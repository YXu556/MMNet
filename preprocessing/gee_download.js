var networks_5 = ee.FeatureCollection("path/to/your/network");
var SSM = networks_5.select('ID', 'Network', 'Latitude', 'Longitude');

// Define a buffer distance in meters
var bufferDistance = 50 * 1.414; // Adjust this value as needed
var res = '0.1'

// Create a new FeatureCollection with buffered points
var bufferedSSM = SSM.map(function(point) {
  return point.buffer(bufferDistance);
});

// ----------------------------------- //
// ------------ Functions ------------ //
// ----------------------------------- //

// add 'date_daily' to image property
function add_date(img){
    var date  = ee.Date(img.get('system:time_start'));
    var date_daily = date.format('YYYYMMdd');
    // var doy = ee.Date(img.get('system:time_start')).getRelative('day','year');
    return img
      // .addBands(ee.Image.constant(doy).rename('doy').float())
      .set('date_daily', date_daily);
}

function maskHLSL30(image) {
  // Bits 3 and 1 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 1);
  // var snowBitMask = (1 << 4);
  // Get the pixel QA band.
  var qa = image.select('Fmask');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                .and(qa.bitwiseAnd(cloudsBitMask).eq(0))
                // .and(qa.bitwiseAnd(snowBitMask).eq(0));
  return image.updateMask(mask);
}

// ***************static features****************************
var dem = ee.Terrain.products(ee.Image("USGS/3DEP/10m"));
var clay_5 = ee.Image("users/xuehaiwuya8/ML-HRSM/polaris_clay_5");
var sand_5 = ee.Image("users/xuehaiwuya8/ML-HRSM/polaris_sand_5");
var bd_5 = ee.Image("users/xuehaiwuya8/ML-HRSM/polaris_bd_5")

var soil_5 = clay_5.rename('clay_5').addBands(sand_5.rename('sand_5')).addBands(bd_5.rename('bd_5'));
var constant = dem.addBands(soil_5).reproject('EPSG:4326', null, 30);
var constant_SSM =  constant.reduceRegions({
  collection: bufferedSSM,
  reducer: ee.Reducer.mean(),
  scale: 30 // resolution
  });

Export.table.toDrive({
  collection: constant_SSM,
  description: 'constant_SSM',
  folder: res,
  fileFormat: 'CSV'
});

// for several years
for (var year_int=2021;year_int<2023;year_int++){
  var year = year_int.toString();
  var start_date = ee.Date.fromYMD(year_int, 1, 1);
  var end_date = ee.Date.fromYMD(year_int, 12, 31);

  // landcover
  var LC = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD').filterBounds(bufferedSSM).filterDate(start_date, end_date).select('landcover');
  var LC = (LC.first().rename('LC'))//.reproject('EPSG:4326', null, 30);

  var LC_SSM = LC.reduceRegions({
    collection: bufferedSSM,
    reducer: ee.Reducer.mode(null, null, 2e5),  // mode for landcover type
    scale: 30
  })
  Export.table.toDrive({
    collection: LC_SSM,
    description: 'LC_SSM_'+year,
    folder: res,
    fileFormat: 'CSV'
  });


  // ***************SMAP soil moisture****************************
  var smap_ssm = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
                  .select('sm_surface').filterBounds(bufferedSSM)
                  .filterDate(start_date,end_date).map(add_date);


  var numberOfDays = end_date.difference(start_date, 'days');

  var smap_ssm_daily = ee.ImageCollection(
    ee.List.sequence(0, numberOfDays.subtract(1))
      .map(function (dayOffset) {
        var start = ee.Date(start_date).advance(dayOffset, 'days');
        var end = start.advance(1, 'days');
        return smap_ssm
          .filterDate(start, end)
          .mean().set('system:index', start.format('YYYYMMdd'));
      })).toBands()//.reproject('EPSG:4326', null, 30).clip(SSM);

  var smap_ssm_SSM =  smap_ssm_daily.reduceRegions({
    collection: bufferedSSM,
    reducer: ee.Reducer.mean(),
    scale: 30 // resolution
    });

  Export.table.toDrive({
    collection: smap_ssm_SSM,
    description: 'smap_ssm_' + year,
    folder: res,
    fileFormat: 'CSV'
  });


  /***********************Gridmet climate data**************************************/
  var gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
                  .select(['pr', 'sph', 'srad', 'vs', 'vpd', 'eto', 'etr'])
                  .filterBounds(bufferedSSM)
                  .filterDate(start_date, end_date)
                  .map(add_date);

  // Map.addLayer(gridmet)
  var gridmet_SSM = gridmet.map(function(image){
    return image.reduceRegions({
    collection: bufferedSSM,
    reducer: ee.Reducer.mean(),
    scale: 30 // resolution of the GRIDMET dataset
    });
  });
  var gridmet_SSM2 = gridmet_SSM.flatten().filter(ee.Filter.neq('pr', null)).select(['.*'],null,false);

  Export.table.toDrive({
    collection: gridmet_SSM2,
    description: 'gridmet_SSM_' + year,
    folder: res,
    fileFormat: 'CSV'
  });


  /***********************Sentinel-1 data**************************************/
  var S1_all = ee.ImageCollection("COPERNICUS/S1_GRD")
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .select(['VV', 'VH', 'angle'])
                .filterBounds(bufferedSSM)
                .filterDate(start_date, end_date)
                .sort('SLC_Processing_start');
  // print(S1_all.limit(5))

  var S1 = S1_all.map(function(img){
      var vv = img.select(['VV']);
      var vv_masked = (vv.mask(vv.gt(-20).and(vv.lt(-5))));
      var vv_filtered = vv_masked.convolve(ee.Kernel.gaussian(3));

      var vh = img.select(['VH']);
      var vh_masked = (vh.mask(vh.gt(-30).and(vh.lt(-10))));
      var vh_filtered = vh_masked.convolve(ee.Kernel.gaussian(3));

      var angle = img.select(['angle']);

      var out = vv_filtered.addBands(vh_filtered).addBands(angle).copyProperties(img);
      return out
    }).map(add_date);

  var S1_SSM = S1.map(function(image){
    return image.reduceRegions({
    collection: bufferedSSM,
    reducer: ee.Reducer.mean(),
    scale: 10 // resolution of the Sentinel-1 dataset
    });
  });

  var S1_SSM2 = S1_SSM.flatten().filter(ee.Filter.neq('VV', null)).select(['.*'], null,false);
  Export.table.toDrive({
      collection: S1_SSM2,
      description: 'S1_SSM_' + year,
      folder: res,
      fileFormat: 'CSV'
    });


  /***********************HLSL30 Data**************************************/    // todo split parts
  var allSamples = bufferedSSM;
  var sampleSize = allSamples.size();
  var allSamples_List = allSamples.toList(sampleSize);
  var parts = 1;
  var interval=ee.Number(sampleSize).divide(parts).toInt();

  for (var i=0; i<parts; i++){
    var start=interval.multiply(i);
    var end=interval.multiply(i+1);
    if(i==(parts-1)){
      end=sampleSize;}

    var currentSamples=ee.FeatureCollection(allSamples_List.slice(start,end));

    var HLSL30 = ee.ImageCollection("NASA/HLS/HLSL30/v002")
                   .map(maskHLSL30)
                   .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9', 'B10', 'B11'])
                   .filterBounds(currentSamples)
                   .filterDate(start_date, end_date).map(add_date);

    var HLSL30_SSM = HLSL30.map(function(image){
      return image.reduceRegions({
      collection: currentSamples,
      reducer: ee.Reducer.mean(),
      scale: 30 // resolution of the HLSL30 dataset
      });
    });
    var HLSL30_SSM2 = HLSL30_SSM.flatten().filter(ee.Filter.neq('B1', null)).select(['.*'],null,false);

    Export.table.toDrive({
      collection: HLSL30_SSM2,
      description: 'HLSL30_SSM_' + year + '_' + (i+1),
      folder: res,
      fileFormat: 'CSV'
    });
  }


  /***********************LST data**************************************/
  function     maskMODIS(image) {
    var    MandatoryBitMask = (1 << 1);
    var    DataBitMask = (1 << 3);
    var    qa = image.select('QC_Day');
    var    mask = qa.bitwiseAnd(MandatoryBitMask).eq(0)
                    .and(qa.bitwiseAnd(DataBitMask).eq(0));

    return    image.updateMask(mask);
      }

  // MODIS daily surface temperature
  var LST = ee.ImageCollection("MODIS/061/MOD11A1")
      .filterBounds(bufferedSSM)
      .filterDate(start_date, end_date)
      .map(maskMODIS)
  		.select('LST_Day_1km');

  var LST_MOD11A1 = LST.map(function(image){
    return image.reduceRegions({
    collection: bufferedSSM,
    reducer: ee.Reducer.mean(),
    scale: 30 // resolution of the GRIDMET dataset
    });
  });
  var LST_MOD11A12 = LST_MOD11A1.flatten().filter(ee.Filter.neq('mean', null))

  Export.table.toDrive({
    collection: LST_MOD11A12,
    description: 'LST_MOD11A1_' + year,
    folder: res,
    fileFormat: 'CSV'
  });
}
