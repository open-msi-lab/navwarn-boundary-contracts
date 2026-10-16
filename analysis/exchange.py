"""GeoJSON encoder for evidence-linked records and a separate consumer-side decoding check."""
import copy, hashlib, json, math
import shapely
from shapely.geometry import shape

KINDS = {'POINT':'Point','MULTIPOINT':'MultiPoint','LINE':'LineString','POLYGON':'Polygon'}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                         allow_nan=False,separators=(',',':')).encode()).hexdigest()

def signed_area(points):
    return sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]))/2

def feature(geom, uid, index, source_sha, record_sha):
    kind = geom['kind']; positions = geom['coordinates']
    if kind not in KINDS:
        raise ValueError('unsupported geometry kind')
    if any(len(p)!=2 or not all(math.isfinite(x) for x in p) or
           abs(p[0])>90 or abs(p[1])>180 for p in positions):
        raise ValueError('invalid source position')
    points = [[lon,lat] for lat,lon in positions]
    if ((kind=='POINT' and len(points)!=1) or
        (kind=='MULTIPOINT' and not points) or
        (kind=='LINE' and len(points)<2) or
        (kind=='POLYGON' and len(points)<3)):
        raise ValueError('invalid geometry arity')
    closed = False; reversed_ring = False
    if kind=='POLYGON' and points[0]!=points[-1]:
        points.append(points[0].copy()); closed=True
    if kind in ('POLYGON','LINE') and any(abs(a[0]-b[0])>180 for a,b in zip(points,points[1:])):
        raise ValueError('antimeridian edge requires a dedicated split')
    if kind=='POLYGON':
        if signed_area(points)<0:
            points=list(reversed(points)); reversed_ring=True
        coordinates=[points]
    elif kind=='POINT':
        coordinates=points[0]
    else:
        coordinates=points
    properties={k:copy.deepcopy(v) for k,v in geom.items() if k not in ('coordinates','kind')}
    properties.update({'warning_uid':uid,'geometry_index':index,'nwie_kind':kind,
                       'source_sha256':source_sha,'record_sha256':record_sha,
                       'adapter_closed_ring':closed,'adapter_reversed_ring':reversed_ring,
                       'width_is_attribute_only':bool(geom.get('radius_nm')),
                       'review_status':'unreviewed'})
    obj={'type':'Feature','id':f'{uid}:geometry:{index}',
         'geometry':{'type':KINDS[kind],'coordinates':coordinates},'properties':properties}
    decoded=shape(obj['geometry'])
    if not decoded.is_valid or decoded.is_empty:
        raise ValueError('invalid planar consumer geometry')
    return obj

def reverse_feature(obj):
    p=obj['properties'];kind=p['nwie_kind'];c=copy.deepcopy(obj['geometry']['coordinates'])
    assert obj['geometry']['type']==KINDS[kind]
    if kind=='POINT':c=[c]
    if kind=='POLYGON':
        assert len(c)==1
        c=c[0]
        if p['adapter_reversed_ring']:c=list(reversed(c))
        if p['adapter_closed_ring']:
            assert c[0]==c[-1];c=c[:-1]
    adapter={'warning_uid','geometry_index','nwie_kind','source_sha256','record_sha256',
             'adapter_closed_ring','adapter_reversed_ring','width_is_attribute_only','review_status'}
    original={k:copy.deepcopy(v) for k,v in p.items() if k not in adapter}
    original.update(kind=kind,coordinates=[[lat,lon] for lon,lat in c])
    return original

def convert(record, text, validator):
    reasons=validator(record,text)
    if reasons:raise ValueError('record validator: '+str(reasons))
    sh=hashlib.sha256(text.encode()).hexdigest()
    if record['text_sha256']!=sh:raise ValueError('source digest mismatch')
    rh=digest(record);uid=record['uid']
    features=[feature(g,uid,i,sh,rh) for i,g in enumerate(record.get('geometry') or [])]
    if not features:
        features=[{'type':'Feature','id':uid+':warning','geometry':None,
                   'properties':{'warning_uid':uid,'source_sha256':sh,'record_sha256':rh,
                                 'geometry_absent':True,'review_status':'unreviewed'}}]
    return {'type':'FeatureCollection','features':features}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def consume(collection, record):
    """Validate decoded geometry and, separately, application identity/properties.

    GEOS reads geometry only. Geometry reconstruction uses its returned object,
    retaining order, repeated vertices and type; no coordinate sets are used.
    """
    from shapely.geometry import mapping
    require(collection.get('type')=='FeatureCollection', 'collection type')
    actual=collection.get('features');originals=record.get('geometry') or []
    require(isinstance(actual,list) and len(actual)==max(1,len(originals)), 'feature count')
    uid=record['uid'];source_sha=record['text_sha256'];record_sha=digest(record)
    if not originals:
        expected={'type':'Feature','id':uid+':warning','geometry':None,
                  'properties':{'warning_uid':uid,'source_sha256':source_sha,
                  'record_sha256':record_sha,'geometry_absent':True,'review_status':'unreviewed'}}
        require(actual[0]==expected, 'null feature identity or properties')
        return 0
    for index,(obj,original) in enumerate(zip(actual,originals)):
        require(obj.get('type')=='Feature' and obj.get('id')==f'{uid}:geometry:{index}', 'feature identity')
        p=obj.get('properties',{})
        expected={k:copy.deepcopy(v) for k,v in original.items() if k not in ('kind','coordinates')}
        expected.update(warning_uid=uid,geometry_index=index,nwie_kind=original['kind'],
            source_sha256=source_sha,record_sha256=record_sha,
            width_is_attribute_only=bool(original.get('radius_nm')),review_status='unreviewed')
        flags={'adapter_closed_ring','adapter_reversed_ring'}
        require(set(p)==set(expected)|flags, 'property keys')
        require(all(type(p[k]) is type(v) and p[k]==v for k,v in expected.items()), 'identity or carried property')
        require(all(type(p[k]) is bool for k in flags), 'normalization flags')
        if original['kind']!='POLYGON':
            require(not any(p[k] for k in flags), 'nonpolygon normalization')
        encoded=obj.get('geometry')
        require(isinstance(encoded,dict) and encoded.get('type')==KINDS[original['kind']], 'encoded type')
        consumer=shapely.from_geojson(json.dumps(encoded))
        require(consumer is not None and consumer.is_valid and not consumer.is_empty, 'consumer validity')
        require(consumer.geom_type==KINDS[original['kind']], 'consumer type')
        # json conversion changes mapping tuples to lists, preserving structure.
        decoded=json.loads(json.dumps(mapping(consumer)))
        rebuilt={'geometry':decoded,'properties':p}
        require(reverse_feature(rebuilt)==original, 'decoded ordered geometry mismatch')
    return len(originals)
