import csv
import torch
import numpy as np
from src.utils.tools import load_pickle
from src.utils.tools import logger
from src.utils.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id,input_len):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.input_len = input_len

class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case,option):
        self.tokenizer = BertTokenizer(vocab_path,do_lower_case)
        self.option = option

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self,lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if self.option == "term":
            return ['platform characteristics', 'atmospheric winds', 'radio wave','weather events', 'geomagnetism', 'atmospheric electricity','microwave', 'atmospheric temperature', 'atmospheric water vapor','atmospheric pressure', 'aerosols', 'atmospheric radiation','atmospheric chemistry', 'precipitation', 'sensor characteristics','radar', 'infrared wavelengths', 'visible wavelengths','weather/climate advisories', 'clouds', 'lidar', 'ocean optics','ultraviolet wavelengths', 'cryospheric indicators','land use/land cover', 'topography', 'surface thermal properties','spectral/engineering', 'soils', 'snow/ice', 'geothermal dynamics','natural hazards', 'surface water', 'vegetation','land surface/agriculture indicators','gravity/gravitational field', 'marine advisories', 'altitude','water quality/water chemistry', 'ocean temperature','ocean winds', 'atmospheric/ocean indicators', 'coastal processes','erosion/sedimentation', 'marine sediments', 'ocean chemistry','salinity/density', 'ocean color', 'aquatic ecosystems','vegetation2', 'landscape', 'cloud properties','surface radiative properties', 'geodetics','agricultural plant science', 'forest science','ecological dynamics', 'environmental impacts', 'sustainability','boundaries', 'ecosystems', 'air quality', 'population','infrastructure', 'environmental governance/management','public health', 'economic resources', 'socioeconomics','environmental vulnerability index (evi)', 'human settlements','agricultural chemicals', 'animal science','habitat conversion/fragmentation', 'animals/vertebrates','earth gases/liquids', 'rocks/minerals/crystals','social behavior', 'ground water', 'frozen ground','terrestrial hydrosphere indicators', 'ocean heat budget','biospheric indicators', 'animal commodities', 'fungi', 'plants','carbon flux', 'geomorphic landforms/processes','paleoclimate indicators', 'ocean circulation', 'sea ice','geochemistry', 'visualization/image processing','subsetting/supersetting', 'transformation/conversion','ocean pressure', 'glaciers/ice sheets', 'protists','solar activity', 'sun-earth interactions','sea surface topography', 'solar energetic particle properties','solar energetic particle flux','ionosphere/magnetosphere dynamics']
        elif self.option == "mostdepth":
            return ['flight data logs', 'turbulence', 'radio wave flux', 'lightning', 'magnetic field', 'atmospheric conductivity', 'electric field', 'data synchronization time', 'brightness temperature', 'vertical profiles', 'water vapor profiles', 'air temperature', 'upper level winds', 'atmospheric pressure measurements', 'upper air temperature', 'humidity', 'dew point temperature', 'aerosol particle properties', 'emissivity', 'trace gases/trace species', 'liquid precipitation', 'cloud liquid water/ice', 'microwave radiance', 'sensor counts', 'total pressure', 'airspeed/ground speed', 'total temperature', 'static pressure', 'wind speed', 'wind direction', 'radar reflectivity', 'doppler velocity', 'infrared imagery', 'visible imagery', 'water vapor', 'vertical wind velocity/speed', 'aerosol backscatter', 'weather forecast', 'tropical cyclones', 'visible radiance', 'infrared radiance', 'total precipitable water', 'boundary layer temperature', 'atmospheric temperature indices', 'cloud height', 'flight level winds', 'cloud droplet distribution', 'cloud droplet concentration/size', 'cloud condensation nuclei', 'cloud microphysics', 'hydrometeors', 'ozone', 'wind profiles', 'cloud base temperature', 'cloud base height', 'liquid water equivalent', 'solar radiation', 'planetary boundary layer height', 'surface winds', 'precipitation amount', 'precipitation rate', 'surface pressure', 'rain', 'cloud optical depth/thickness', 'aerosol extinction', 'aerosol optical depth/thickness', 'cirrus cloud systems', 'lidar depolarization ratio', 'radar backscatter', 'radar cross-section', 'return power', 'mean radial velocity', 'radiance', 'air quality', 'climate advisories', 'atmospheric emitted radiation', 'optical depth/thickness', 'surface temperature', 'ultraviolet flux', 'spectrum width', 'microwave imagery', 'lidar backscatter', 'relative humidity', 'u/v wind components', 'wind speed/wind direction', 'radar imagery', 'snow depth', 'land use/land cover classification', 'digital elevation/terrain model (dem)', 'snow', 'droplet size', 'droplet concentration/size', 'drizzle', 'precipitation anomalies', 'snow water equivalent', 'solid precipitation', 'total surface precipitation rate', 'particle size distribution', 'skin temperature', 'attitude characteristics', 'land surface temperature', 'hail', 'reflectance', 'soil moisture/water content', 'soil temperature', 'soil bulk density', 'surface roughness', 'present weather', 'snow density', 'ambient temperature', 'aerosol forward scatter', 'floods', 'snow cover', 'sigma naught', 'precipitable water', 'stage height', 'rivers/streams', 'shortwave radiation', 'photosynthetically active radiation', 'longwave radiation', 'net radiation', 'hourly precipitation amount', '24 hour precipitation amount', 'soil moisture', 'satellite orbits/revolution', 'sea surface temperature', 'heat flux', 'latent heat flux', 'cloud fraction', '3 and 6 hour precipitation amount', 'geopotential height', 'particulate matter', 'particle images', 'water vapor indices', 'horizontal wind velocity/speed', 'electrical conductivity', 'dissolved carbon dioxide', 'hurricanes', 'tropical cyclone track', 'convective clouds/systems (observed/analyzed)', 'cloud top height', 'viewing geometry', 'temperature profiles', 'vertical wind shear', 'wind shear', 'carbon monoxide', 'sea level pressure', 'water vapor tendency', 'potential temperature', 'angstrom exponent', 'ultraviolet radiation', 'solar irradiance', 'scattering', 'absorption', 'water vapor mixing ratio profiles', 'sea surface temperature indices', 'extreme eastern tropical pacific sst', 'sedimentation', 'erosion', 'sediment transport', 'sediments', 'tropopause', 'ocean chemistry', 'ocean optics', 'ocean temperature', 'salinity/density', 'pigments', 'ocean color', 'attenuation/transmission', 'inorganic carbon', 'organic carbon', 'photosynthetically available radiation', 'chlorophyll', 'optical depth', 'fluorescence', 'vegetation index', 'gelbstoff', 'phytoplankton', 'vegetation index2', 'cloud precipitable water', 'landscape ecology', 'ultraviolet radiance', 'cloud ceiling', 'aerosol radiance', 'carbonaceous aerosols', 'dust/ash/smoke', 'nitrate particles', 'organic particles', 'sulfate particles', 'radiative flux', 'transmittance', 'atmospheric stability', 'cloud asymmetry', 'cloud frequency', 'cloud top pressure', 'cloud top temperature', 'cloud vertical distribution', 'cloud emissivity', 'cloud radiative forcing', 'cloud reflectance', 'rain storms', 'reflected infrared', 'thermal infrared', 'incoming solar radiation', 'clouds', 'cloud properties', 'cloud types', 'orbital characteristics', 'sensor characteristics', 'maximum/minimum temperature', 'condensation', 'platform characteristics', 'geolocation', 'geodetics', 'coordinate reference system', 'aerosols', 'topographical relief maps', 'terrain elevation', 'normalized difference vegetation index (ndvi)', 'infrared flux', 'visible flux', 'albedo', 'land use/land cover', 'topography', 'lidar', 'lidar waveform', 'plant phenology', 'vegetation cover', 'crop/plant yields', 'land use classes', 'landscape patterns', 'forest harvesting and engineering', 'forest management', 'total surface water', 'agricultural plant science', 'photosynthesis', 'primary production', 'leaf characteristics', 'evapotranspiration', 'fire occurrence', 'surface thermal properties', 'canopy characteristics', 'evergreen vegetation', 'crown', 'deciduous vegetation', 'anisotropy', 'fire ecology', 'biomass burning', 'wildfires', 'topographical relief', 'burned area', 'surface radiative properties', 'environmental sustainability', 'boundaries', 'anthropogenic/human influenced ecosystems', 'emissions', 'sulfur dioxide', 'population', 'infrastructure', 'environmental assessments', 'public health', 'conservation', 'agriculture production', 'administrative divisions', 'economic resources', 'socioeconomics', 'lake/pond', 'rivers/stream', 'political divisions', 'environmental vulnerability index (evi)', 'ecosystems', 'urban areas', 'sustainability', 'treaty agreements/results', 'human settlements', 'population estimates', 'nitrogen dioxide', 'cropland', 'pasture', 'particulates', 'cyclones', 'mortality', 'environmental impacts', 'droughts', 'earthquakes', 'population distribution', 'fertilizers', 'animal manure and waste', 'urbanization/urban sprawl', 'landslides', 'avalanche', 'urban lands', 'mangroves', 'volcanic eruptions', 'pesticides', 'population size', 'population density', 'lakes/reservoirs', 'surface water', 'rural areas', 'infant mortality rates', 'amphibians', 'mammals', 'carbon', 'sulfur oxides', 'methane', 'non-methane hydrocarbons/volatile organic compounds', 'nitrogen oxides', 'natural gas', 'coal', 'coastal elevation', 'biodiversity functions', 'nuclear radiation exposure', 'radiation exposure', 'poverty levels', 'malnutrition', 'wetlands', 'sea level rise', 'vulnerability levels/index', 'ground water', 'snow/ice', 'electricity', 'energy production/use', 'sustainable development', 'deforestation', 'household income', 'discharge/flow', 'hydropattern', 'nitrogen', 'phosphorus', 'carbon dioxide', 'alpine/tundra', 'forests', 'vegetation', 'permafrost', 'nutrients', 'plant characteristics', 'leaf area index (lai)', 'soil gas/air', 'ammonia', 'nitrous oxide', 'ecosystem functions', 'litter characteristics', 'soil chemistry', 'soil respiration', 'active layer', 'soil depth', 'cation exchange capacity', 'organic matter', 'soil porosity', 'soil texture', 'permafrost melt', 'land subsidence', 'freeze/thaw', 'surface water features', 'chlorinated hydrocarbons', 'methyl bromide', 'methyl chloride', 'molecular hydrogen', 'sulfur compounds', 'fire models', 'biomass', 'dominant species', 'vegetation species', 'sulfur', 'tree rings', 'soil classification', 'heat index', 'sea ice concentration', 'ocean heat budget', 'reforestation', 'even-toed ungulates', 'species recruitment', 'population dynamics', 'range changes', 'topographic effects', 'land resources', 'river ice depth/extent', 'snow melt', 'river ice', 'animal commodities', 'animal ecology and behavior', 'phenological changes', 'water depth', 'inundation', 'forest fire science', 'biogeochemical cycles', 'radiative forcing', 'soil heat budget', 'drainage', 'respiration rate', 'river/lake ice breakup', 'river/lake ice freeze', 'reclamation/revegetation/restoration', 'permafrost temperature', 'indigenous/native species', 'fire dynamics', 'lichens', 'plants', 'plant succession', 'carbon flux', 'coastal', 'salt marsh', 'degradation', 'altitude', 'carbon and hydrocarbon compounds', 'halocarbons and halogens', 'forest composition/vegetation structure', 'water vapor indicators', 'barometric altitude', 'atmospheric water vapor', 'terrestrial ecosystems', 'volatile organic compounds', 'boundary layer winds', 'forest fire danger index', 'periglacial processes', 'landscape processes', 'evaporation', 'soil horizons/profile', 'shrubland/scrub', 'soil ph', 'soils', 'soil water holding capacity', 'community structure', 'pingo', 'soil color', 'virtual temperature', 'formaldehyde', 'hydroxyl', 'photolysis rates', 'cloud dynamics', 'nitric oxide', 'molecular oxygen', 'smog', 'peroxyacyl nitrate', 'hydrogen compounds', 'nitrogen compounds', 'oxygen compounds', 'stable isotopes', 'chemical composition', 'actinic flux', 'tropospheric ozone', 'fossil fuel burning', 'industrial emissions', 'denitrification rate', 'sunshine', 'runoff', 'soil structure', 'mosses/hornworts/liverworts', 'peatlands', 'hydraulic conductivity', 'snow/ice temperature', 'vegetation water content', 'discharge', 'chlorophyll concentrations', 'outgoing longwave radiation', 'geomorphic landforms/processes', 'soil compaction', 'soil impedance', 'canopy transmittance', 'water table', 'decomposition', 'water temperature', 'dissolved gases', 'total dissolved solids', 'agricultural expansion', 'forest science', 'pressure tendency', 'visibility', 'biomass dynamics', 'agricultural lands', 'grasslands', 'savannas', 'grazing dynamics/plant herbivory', 'herbivory', 'paleoclimate reconstructions', 'drought indices', 'fire weather index', 'animal yields', 'multivariate enso index', 'dissolved solids', 'ocean currents', 'salinity', 'coastal processes', 'atmospheric pressure', 'afforestation/reforestation', 'fresh water river discharge', 'surface water chemistry', 'drainage basins', 'resource development site', 'dunes', 'flood plain', 'endangered species', 'precipitation indices', 'temperature indices', 'forest yields', 'stratigraphic sequence', 'freeze/frost', 'frost', 'hydrogen cyanide', 'land management', 'nutrient cycling', 'industrialization', 'suspended solids', 'deserts', 'weathering', 'gas flaring', 'atmospheric temperature', 'ice extent', 'fraction of absorbed photosynthetically active radiation (fapar)', 'marshes', 'swamps', 'lake ice', 'atmospheric winds', 'watershed characteristics', 'transportation', 'soil rooting depth', 'isotopes', 'cultural features', 'consumer behavior', 'boundary surveys', 'aquifers', 'land productivity', 'water quality/water chemistry', 'sediment composition', 'dissolved oxygen', 'surface water processes/measurements', 'turbidity', 'conductivity', 'ph', 'calcium', 'magnesium', 'potassium', 'micronutrients/trace elements', 'social behavior', 'sulfate', 'sediment chemistry', 'biogeochemical processes', 'water ion concentrations', 'cropping systems', 'percolation', 'groundwater chemistry', 'reforestation/revegetation', 'species/population interactions', 'soil infiltration', 'alkalinity', 'soil fertility', 'phosphorous compounds', 'radioisotopes', 'cooling degree days', 'angiosperms (flowering plants)', 'glacial landforms', 'glacial processes', 'contour maps', 'estuaries', 'methane production/use', 'natural gas production/use', 'petroleum production/use', 'visualization/image processing', 'subsetting/supersetting', 'transformation/conversion', 'forest mensuration', 'acid deposition', 'differential pressure', 'precipitation', 'marine ecosystems', 'consumption rates', 'radio wave', 'soil organic carbon (soc)', 'soil erosion', 'halocarbons', 'trace elements/trace metals', 'biomass energy production/use', 'riparian wetlands', 'soil consistence', 'snow stratigraphy', 'thermal conductivity', 'estuary', 'tidal height', 'plant diseases/disorders/pests', 'layered precipitable water', 'atmospheric chemistry', 'water vapor concentration profiles', 'specific humidity', 'total runoff', 'pressure thickness', 'wind stress', 'atmospheric heating', 'conduction', 'hydrogen chloride', 'nitric acid', 'radar', 'land surface/agriculture indicators', 'satellite soil moisture index', 'chlorine nitrate', 'chlorofluorocarbons', 'dinitrogen pentoxide', 'antenna temperature', 'glaciers', 'ice sheets', 'dimethyl sulfide', 'potential vorticity', 'ice fraction', 'atmospheric radiation', 'runoff rate', 'temperature tendency', 'wind dynamics', 'wind direction tendency', 'base flow', 'bromine monoxide', 'chlorine monoxide', 'methyl cyanide', 'hypochlorous acid', 'methanol', 'hydroperoxy', 'cloud base pressure', 'temperature anomalies', 'nitrate', 'ocean mixed layer', 'precipitation trends', 'temperature trends', 'convection', 'ground ice', 'oxygen', 'phosphate', 'solar induced fluorescence', 'chlorine dioxide', 'sun-earth interactions', 'uv aerosol index', 'volcanic activity', 'potential evapotranspiration', 'ultraviolet wavelengths', 'ice temperature', 'sea surface skin temperature', 'sea surface height', 'sublimation', 'convective surface precipitation rate', 'hydrogen fluoride', 'airglow', 'energy deposition', 'x-ray flux', 'electron flux', 'proton flux', 'magnetic fields/magnetic currents']
        else:
            return ['platform characteristics', 'atmospheric winds','radio wave', 'weather events', 'geomagnetism','atmospheric electricity', 'microwave', 'atmospheric temperature','atmospheric water vapor', 'atmospheric pressure', 'aerosols','atmospheric radiation', 'atmospheric chemistry', 'precipitation','sensor characteristics', 'radar', 'infrared wavelengths','visible wavelengths', 'weather/climate advisories', 'clouds','lidar', 'ocean optics', 'ultraviolet wavelengths','cryospheric indicators', 'land use/land cover', 'topography','surface thermal properties', 'spectral/engineering', 'soils','snow/ice', 'geothermal dynamics', 'natural hazards','surface water', 'vegetation','land surface/agriculture indicators','gravity/gravitational field', 'marine advisories', 'altitude','water quality/water chemistry', 'ocean temperature','ocean winds', 'atmospheric/ocean indicators', 'coastal processes','erosion/sedimentation', 'marine sediments', 'ocean chemistry','salinity/density', 'ocean color', 'aquatic ecosystems','vegetation2', 'landscape', 'cloud properties','surface radiative properties', 'geodetics','agricultural plant science', 'forest science','ecological dynamics', 'environmental impacts', 'sustainability','boundaries', 'ecosystems', 'air quality', 'population','infrastructure', 'environmental governance/management','public health', 'economic resources', 'socioeconomics','environmental vulnerability index (evi)', 'human settlements','agricultural chemicals', 'animal science','habitat conversion/fragmentation', 'animals/vertebrates','earth gases/liquids', 'rocks/minerals/crystals','social behavior', 'ground water', 'frozen ground','terrestrial hydrosphere indicators', 'ocean heat budget','biospheric indicators', 'animal commodities', 'fungi', 'plants','carbon flux', 'geomorphic landforms/processes','paleoclimate indicators', 'ocean circulation', 'sea ice','geochemistry', 'visualization/image processing','subsetting/supersetting', 'transformation/conversion','ocean pressure', 'glaciers/ice sheets', 'protists','solar activity', 'sun-earth interactions','sea surface topography', 'solar energetic particle properties','solar energetic particle flux','ionosphere/magnetosphere dynamics','flight data logs','wind dynamics', 'radio wave flux', 'lightning', 'magnetic field','atmospheric conductivity', 'electric field','data synchronization time', 'brightness temperature','upper air temperature', 'water vapor profiles','surface temperature', 'upper level winds','atmospheric pressure measurements', 'water vapor indicators','aerosol particle properties', 'emissivity','trace gases/trace species', 'liquid precipitation','cloud microphysics', 'microwave radiance', 'sensor counts','total pressure', 'airspeed/ground speed', 'total temperature','static pressure', 'humidity', 'radar reflectivity','doppler velocity', 'infrared imagery', 'visible imagery','aerosol backscatter', 'weather forecast', 'tropical cyclones','visible radiance', 'infrared radiance','atmospheric temperature indices', 'cloud droplet distribution','cloud condensation nuclei', 'hydrometeors', 'oxygen compounds','wind profiles', 'liquid water equivalent', 'solar radiation','planetary boundary layer height', 'surface winds','precipitation amount', 'precipitation rate', 'surface pressure','aerosol extinction', 'aerosol optical depth/thickness','tropospheric/high-level clouds (observed/analyzed)','lidar depolarization ratio', 'radar backscatter','radar cross-section', 'return power', 'radial velocity','radiance', 'climate advisories', 'atmospheric emitted radiation','optical depth/thickness', 'ultraviolet flux', 'spectrum width','microwave imagery', 'lidar backscatter', 'radar imagery','snow depth', 'land use/land cover classification','terrain elevation', 'solid precipitation', 'droplet size','droplet concentration/size', 'precipitation anomalies','snow water equivalent', 'total surface precipitation rate','skin temperature', 'water vapor', 'attitude characteristics','land surface temperature', 'reflectance','soil moisture/water content', 'soil temperature','soil bulk density', 'surface roughness', 'present weather','snow density', 'geothermal temperature','aerosol forward scatter', 'floods', 'snow cover', 'sigma naught','precipitable water', 'surface water processes/measurements','surface water features', 'shortwave radiation','photosynthetically active radiation', 'longwave radiation','net radiation', 'flight level winds', 'soil moisture','satellite orbits/revolution', 'heat flux','precipitation profiles', 'geopotential height','particulate matter', 'particle images', 'water vapor indices','electrical conductivity', 'gases', 'sea surface temperature','convective clouds/systems (observed/analyzed)','viewing geometry', 'wind shear','carbon and hydrocarbon compounds', 'sea level pressure','water vapor processes', 'ultraviolet radiation','solar irradiance', 'scattering', 'absorption','sea surface temperature indices', 'sedimentation', 'erosion','sediment transport', 'sediments', 'tropopause', 'nan', 'pigments','attenuation/transmission', 'inorganic carbon', 'organic carbon','photosynthetically available radiation', 'chlorophyll','optical depth', 'fluorescence', 'vegetation index', 'gelbstoff','plankton', 'vegetation index2', 'landscape ecology','ultraviolet radiance', 'aerosol radiance','carbonaceous aerosols', 'dust/ash/smoke', 'nitrate particles','organic particles', 'sulfate particles', 'radiative flux','transmittance', 'atmospheric stability','cloud radiative transfer', 'rain storms', 'reflected infrared','thermal infrared', 'incoming solar radiation', 'cloud types','orbital characteristics', 'geolocation','coordinate reference system', 'infrared flux', 'visible flux','albedo', 'lidar waveform', 'plant phenology', 'vegetation cover','crop/plant yields', 'land use classes', 'landscape patterns','forest harvesting and engineering', 'forest management','ecosystem functions', 'leaf characteristics', 'fire ecology','total surface water', 'primary production', 'photosynthesis','canopy characteristics', 'evergreen vegetation', 'crown','deciduous vegetation', 'anisotropy', 'biomass burning','wildfires', 'topographical relief','environmental sustainability','anthropogenic/human influenced ecosystems', 'emissions','sulfur compounds', 'environmental assessments', 'conservation','agriculture production', 'administrative divisions','freshwater ecosystems', 'political divisions', 'urban areas','treaty agreements/results', 'population estimates','nitrogen compounds', 'particulates', 'mortality', 'droughts','earthquakes', 'population distribution', 'fertilizers','animal manure and waste', 'urbanization/urban sprawl','landslides', 'avalanche', 'mangroves', 'volcanic eruptions','pesticides', 'population size', 'population density','rural areas', 'amphibians', 'mammals', 'carbon', 'sulfur oxides','land management', 'natural gas', 'sedimentary rocks','coastal elevation', 'community dynamics','nuclear radiation exposure', 'radiation exposure','poverty levels', 'malnutrition', 'sea level rise','vulnerability levels/index', 'electricity','energy production/use', 'sustainable development','deforestation', 'household income', 'nitrogen', 'phosphorus','terrestrial ecosystems', 'permafrost', 'nutrients','plant characteristics', 'soil gas/air', 'litter characteristics','soil chemistry', 'soil respiration', 'active layer', 'soil depth','cation exchange capacity', 'organic matter', 'soil porosity','soil texture', 'permafrost melt','ground water processes/measurements', 'freeze/thaw','halocarbons and halogens', 'hydrogen compounds', 'biomass','dominant species', 'vegetation species', 'sulfur', 'tree rings','soil classification', 'sea ice concentration', 'reforestation','species/population interactions', 'range changes','topographic effects', 'land resources', 'river ice depth/extent','snow melt', 'river ice', 'animal ecology and behavior','phenological changes', 'forest fire science', 'radiative forcing','soil heat budget', 'river/lake ice breakup','river/lake ice freeze', 'reclamation/revegetation/restoration','lichens', 'marine ecosystems', 'coastal landforms', 'degradation','forest composition/vegetation structure', 'barometric altitude','volatile organic compounds', 'forest fire danger index','periglacial processes', 'landscape processes','soil horizons/profile', 'soil ph', 'soil water holding capacity','fluvial landforms', 'soil color', 'glacial processes','photochemistry', 'cloud dynamics', 'nitrogen oxides', 'smog','chemical composition', 'actinic flux', 'tropospheric ozone','fossil fuel burning', 'industrial emissions','denitrification rate', 'sunshine', 'soil structure','mosses/hornworts/liverworts', 'hydraulic conductivity','snow/ice temperature', 'water characteristics','outgoing longwave radiation', 'soil compaction', 'soil impedance','canopy transmittance', 'ground water features', 'solids','agricultural expansion', 'pressure tendency', 'visibility','herbivory', 'paleoclimate reconstructions', 'drought indices','fire weather index', 'animal yields', 'teleconnections','carbon dioxide', 'dissolved solids', 'ocean currents', 'salinity','afforestation/reforestation', 'fresh water river discharge','surface water chemistry', 'aeolian landforms','precipitation indices', 'temperature indices', 'forest yields','stratigraphic sequence', 'freeze/frost', 'frost','industrialization', 'ice core records', 'suspended solids','weathering', 'gas flaring', 'ice extent', 'biogeochemical cycles','lake ice', 'isotopes', 'watershed characteristics','transportation', 'soil rooting depth', 'geochemical properties','carbon monoxide', 'cultural features', 'consumer behavior','boundary surveys', 'land productivity', 'sediment composition','calcium', 'magnesium', 'potassium','micronutrients/trace elements', 'sediment chemistry','biogeochemical processes', 'cropping systems','groundwater chemistry', 'reforestation/revegetation','soil infiltration', 'soil fertility','angiosperms (flowering plants)', 'glacial landforms','forest mensuration', 'acid deposition', 'differential pressure','soil erosion', 'trace elements/trace metals', 'soil consistence','snow stratigraphy', 'thermal conductivity', 'estuaries','tidal height', 'plant diseases/disorders/pests','pressure thickness', 'atmospheric heating', 'conduction','evaporation', 'turbulence', 'wind stress','satellite soil moisture index', 'antenna temperature', 'glaciers','ice sheets', 'nitrate', 'ocean mixed layer','precipitation indicators', 'temperature indicators', 'ground ice','alkalinity', 'dissolved gases', 'oxygen', 'ph', 'phosphate','solar induced fluorescence', 'volcanic activity','ice temperature', 'sea surface height', 'airglow','energy deposition', 'x-ray flux', 'electron flux', 'proton flux','magnetic fields/magnetic currents', 'vertical profiles','air temperature', 'dew point temperature','cloud liquid water/ice', 'wind speed', 'wind direction','vertical wind velocity/speed', 'total precipitable water','boundary layer temperature', 'cloud height','cloud droplet concentration/size', 'ozone','cloud base temperature', 'cloud base height', 'rain','cloud optical depth/thickness', 'cirrus/systems','mean radial velocity', 'relative humidity', 'u/v wind components','wind speed/wind direction','digital elevation/terrain model (dem)', 'snow', 'drizzle','particle size distribution', 'hail', 'ambient temperature','stage height', 'rivers/streams', 'hourly precipitation amount','24 hour precipitation amount', 'latent heat flux','cloud fraction', '3 and 6 hour precipitation amount','horizontal wind velocity/speed', 'dissolved carbon dioxide','hurricanes', 'tropical cyclone track', 'cloud top height','temperature profiles', 'vertical wind shear','water vapor tendency', 'potential temperature','angstrom exponent', 'water vapor mixing ratio profiles','extreme eastern tropical pacific sst', 'phytoplankton','cloud precipitable water', 'cloud asymmetry', 'cloud ceiling','cloud frequency', 'cloud top pressure', 'cloud top temperature','cloud vertical distribution', 'cloud emissivity','cloud radiative forcing', 'cloud reflectance','maximum/minimum temperature', 'condensation','topographical relief maps', 'evapotranspiration','fire occurrence', 'burned area', 'sulfur dioxide', 'lake/pond','rivers/stream', 'nitrogen dioxide', 'agricultural lands','cyclones', 'urban lands', 'lakes/reservoirs','infant mortality rates', 'methane','non-methane hydrocarbons/volatile organic compounds', 'coal','biodiversity functions', 'wetlands', 'discharge/flow','hydropattern', 'alpine/tundra', 'forests','leaf area index (lai)', 'ammonia', 'nitrous oxide','land subsidence', 'normalized difference vegetation index (ndvi)','chlorinated hydrocarbons', 'methyl bromide', 'methyl chloride','molecular hydrogen', 'fire models', 'heat index','even-toed ungulates', 'species recruitment','population dynamics', 'water depth', 'inundation', 'drainage','respiration rate', 'permafrost temperature','indigenous/native species', 'fire dynamics', 'plant succession','coastal', 'salt marsh', 'boundary layer winds', 'shrubland/scrub','community structure', 'pingo', 'virtual temperature','formaldehyde', 'hydroxyl', 'photolysis rates', 'nitric oxide','molecular oxygen', 'peroxyacyl nitrate', 'stable isotopes','runoff', 'vegetation water content', 'discharge','chlorophyll concentrations', 'water table', 'decomposition','water temperature', 'total dissolved solids', 'biomass dynamics','grasslands', 'savannas', 'grazing dynamics/plant herbivory','multivariate enso index', 'drainage basins','resource development site', 'dunes', 'flood plain','endangered species', 'hydrogen cyanide', 'nutrient cycling','deserts','fraction of absorbed photosynthetically active radiation (fapar)','aquifers', 'dissolved oxygen', 'turbidity', 'conductivity','sulfate', 'water ion concentrations', 'percolation','phosphorous compounds', 'radioisotopes', 'cooling degree days','contour maps', 'methane production/use','natural gas production/use', 'petroleum production/use','consumption rates', 'soil organic carbon (soc)', 'halocarbons','biomass energy production/use', 'estuary','layered precipitable water', 'water vapor concentration profiles','hydrogen chloride', 'nitric acid', 'chlorine nitrate','chlorofluorocarbons', 'dinitrogen pentoxide', 'dimethyl sulfide','vorticity', 'ice fraction', 'temperature tendency','wind direction tendency', 'bromine monoxide', 'chlorine monoxide','methyl cyanide', 'hypochlorous acid', 'methanol', 'hydroperoxy','cloud base pressure', 'temperature anomalies','precipitation trends', 'temperature trends', 'convection','chlorine dioxide', 'uv aerosol index','sea surface skin temperature', 'sublimation','convective surface precipitation rate', 'hydrogen fluoride']

    @classmethod
    def read_data(cls, input_file,quotechar = None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self,lines,example_type,cached_examples_file):
        '''
        Creates examples for data
        '''
        pbar = ProgressBar(n_total = len(lines),desc='create examples')
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i,line in enumerate(lines):
                guid = '%s-%d'%(example_type,i)
                text_a = line[0]
                label = line[1]
                if isinstance(label,str):
                    label = [np.float(x) for x in label.split(",")]
                else:
                    label = [np.float(x) for x in list(label)]
                text_b = None
                example = InputExample(guid = guid,text_a = text_a,text_b=text_b,label= label)
                examples.append(example)
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self,examples,max_seq_len,cached_features_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        pbar = ProgressBar(n_total=len(examples),desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id,example in enumerate(examples):
                tokens_a = self.tokenizer.tokenize(example.text_a)
                tokens_b = None
                label_id = example.label

                if example.text_b:
                    tokens_b = self.tokenizer.tokenize(example.text_b)
                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    self.truncate_seq_pair(tokens_a,tokens_b,max_length = max_seq_len - 3)
                else:
                    # Account for [CLS] and [SEP] with '-2'
                    if len(tokens_a) > max_seq_len - 2:
                        tokens_a = tokens_a[:max_seq_len - 2]
                tokens = ['[CLS]'] + tokens_a + ['[SEP]']
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ['[SEP]']
                    segment_ids += [1] * (len(tokens_b) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_len - len(input_ids))
                input_len = len(input_ids)

                input_ids   += padding
                input_mask  += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")

                feature = InputFeature(input_ids = input_ids,
                                       input_mask = input_mask,
                                       segment_ids = segment_ids,
                                       label_id = label_id,
                                       input_len = input_len)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self,features,is_sorted = False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(features,key=lambda x:x.input_len,reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features],dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens)
        return dataset

