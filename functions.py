import glob, os, tqdm, math, copy, cv2, cf, csv
import multiprocessing as mp
from geopy.distance import geodesic
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

def load_dl_data(path: str = "/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered", years: list[int] = None, whole_world: bool = True) -> tuple(np.array, np.array):

    '''
    Function to load and return all data needed for DL testing, i.e. between 1st August 2017 and 31st August 2019
    
    args:
        path: path to save location
        years: list of years to consider for experiments
        whole_world: boolean to decide whether to use all regions or just WAWP regions
    returns:
        data: cases from opened files as one numpy array; in the same order as files
        labels: label for each case in data
    '''
    
    # Get list of all files
    all_files=list(sorted(glob.glob(os.path.join(path, "*.npz"))))
    
    if whole_world == False:
        files = []
        for file in all_files:
            if "0_60_260_340" in file or "0_60_100_180" in file:
                files.append(file)
        all_files = files
        
    if years != None:
        files = []
        for file in all_files:
            name = file.split("/")[-1].replace(".npz", "")
            date = name.split("_")[-1]
            year = int(date[:4])
            if year in years:
                files.append(file)
        all_files = files
    
    # Open pool for multiprocessing
    pool = mp.Pool(int(0.5*mp.cpu_count()))
    
    # Get cases
    results = list(tqdm.tqdm(pool.imap(get_dl_file, all_files), total=len(all_files)))
    
    # Close pool after all cases loaded
    pool.close()
    
    # Set up numpy array to hold all cases
    shape = results[0][0].shape
    
    data = np.zeros((len(results), shape[0], shape[1], shape[2]))
    labels = np.zeros(len(results))
    
    # Loop through all cases and place in numpy array holding all cases
    for i, res in enumerate(results):
        data[i], labels[i] = res
    
    # Return the list of files opened and their case data
    return data, labels, all_files

def rebin(a: np.array, shape: list[int], dim: int) -> np.array:

    '''
    Function to resize (reduce) an array to a given shape
    args:
        a: 3D array to be shaped
        shape: 2D shape of new array
        dim: number of channels [== a.shape[-1]]
    returns:
        new_array: a in the shape needed
    '''

    sh = shape[0],dim,shape[1],dim
    return a.reshape(sh).mean(-1).mean(1)


def reduce_dim(dim: int, big_arr: np.array) -> np.array:

    '''
    Function to reduce an array to a given 1D factor
    args:
        dim: factor by which the array is to be resized
        big_arr: array to be reduced in size
    returns
        small_arr: reduced array
    '''
    
    # if dim == 1, then no need to resize
    if dim == 1:
        return big_arr
    
    # pad array to get right shape for resizing
    if dim == 3:
        big_arr = np.pad(big_arr, ((0, 4), (0, 0), (0, 0))) 
    elif dim == 4:
        big_arr = np.pad(big_arr, ((0, 2), (0, 2), (0, 0))) 
    elif dim == 5:
        big_arr = np.pad(big_arr, ((0, 4), (0, 1), (0, 0))) 
    
    # compute shape of resized array
    small_shape = (big_arr.shape[0]//dim, big_arr.shape[1]//dim, big_arr.shape[2])

    # create resized array, initilized as zeros
    small_arr = np.zeros(small_shape)

    # resize each channel and place in array
    for i in range(big_arr.shape[-1]):
        small_arr[:,:,i] = rebin(big_arr[:,:,i], small_shape[0:-1], dim)

    return small_arr

def get_dl_file(file: str) -> tuple(np.array, int):

    '''
    Function to load a single case for DL testing
    args:
        file: path of file to load
    returns:
        arr: case that file holds, in a numpy array of shape (22, 29, 5)
    '''    
    
    # Get file
    arr = np.load(file)['arr_0']
    
    # Reduce to a sixteenth (4*4) of ERAI resolution
    arr = reduce_dim(4, arr)
        
    # Standardisation
    for field in range(arr.shape[-1]):
        mean_ = np.mean(arr[:,:,field])
        std_ = np.std(arr[:,:,field])
        arr[:,:,field] = (arr[:,:,field] - mean_) / std_
    
    # get label based on file name
    name = file.split("/")[-1].replace(".npz", "")
    cat = name.split("_")[0]
    if cat == "no":
        label = 0
    else:
        cat = int(cat)
        if cat < 1:
            label = 0
        else:
            label = 1
    
    return arr, label

def load_nc_file(inputs: list[list[dt, float, float, float]]) -> tuple(np.array, np.array, np.array):

    '''
    Function to load data from original ERAI data with a given lat/lon in the centre of the data
    args:
        inputs: list of inputs; needed for multiprocessing to work
            date: date of case
            lat: lat to have in the centre of the data
            lon: lon to have in the centre of the data
            extent: length of bounding box for data, centred at lat/lon
    returns:
        array: numpy array holding data needed
        lats: list of latitudes for case
        lons: list of longitudes for case
    '''
    
    # Load inputs
    date, lat, lon, extent = inputs
    
    # Path for ERAI file holding MSLP, wind data 
    load_path = "/badc/ecmwf-era-interim/data/gg/as/" + str(date.year) + "/" + str(date.month).zfill(2) + "/" + str(date.day).zfill(2) + "/ggas" + str(date.year).zfill(2) + str(date.month).zfill(2) + str(date.day).zfill(2) + str(date.hour).zfill(2) + "00.nc"

    # Reset lon to [0, 360] range
    if lon < 0:
        lon += 360
        
    # Calculate bounding box needed
    min_lat = lat - extent/2.
    max_lat = lat + extent/2.
    min_lon = lon - extent/2.
    max_lon = lon + extent/2.
    
    # Load ERAI file
    nc = cf.read(load_path)

    # Get list lof lats/lons for bounding box
    init_field = nc[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat))
    lats = init_field.constructs['dimensioncoordinate2'].data.array
    lons = init_field.constructs['dimensioncoordinate3'].data.array
    
    # Get MSLP field
    mslp = nc.select_by_identity("air_pressure_at_sea_level")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat)).data.array[0, 0, :, :]

    # Get U field
    u = nc.select_by_identity("northward_wind").select_by_units("m s**-1")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat)).data.array[0, 0, :, :]
    
    # Get V field
    v = nc.select_by_identity("eastward_wind").select_by_units("m s**-1")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat)).data.array[0, 0, :, :]
    
    # Calculate wind speed
    wind = np.sqrt(u**2 + v**2)

    # Close ERAI file
    nc.close()

    # Open ERAI file holding vorticity fields
    nc = cf.read(load_path.replace("as", "ap"))

    # Load vorticity
    vort850 = nc.select_by_identity("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat), Z=850).data.array[0, 0, :, :]
    vort700 = nc.select_by_identity("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat), Z=700).data.array[0, 0, :, :]
    vort600 = nc.select_by_identity("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(X=cf.wi(min_lon, max_lon), Y=cf.wi(min_lat, max_lat), Z=600).data.array[0, 0, :, :]

    # Set up numpy array to hold the data
    array = np.zeros((mslp.shape[0], mslp.shape[1], 5))
    
    # Add MSLP data
    array[:, :, 0] = mslp
    
    # Add wind speed data
    array[:, :, 1] = wind
    
    # Add vorticity data; flip sign if in Southern Hemisphere 
    if (min_lat + max_lat) / 2 < 0:
        array[:, :, 2] = -vort850
        array[:, :, 3] = -vort700
        array[:, :, 4] = -vort600
    else:
        array[:, :, 2] = vort850
        array[:, :, 3] = vort700
        array[:, :, 4] = vort600
        
    return array, lats, lons

def get_heatmap(model: tf.keras.Model, layer_name: str, x: np.array) -> np.array:

    '''
    Function to get a heatmap for class activations from a given DL CNN-based model at a given layer for given data
    args:
        model: DL CNN-based model
        layer_name: name for layer present in model
        x: data for which generate heatmap
    returns:
        heatmap: 2D numpy array containing heatmap 
    '''
    
    # Preprocessing of data
    x = np.expand_dims(x, axis=0)
    
    # Get layer at which to generate heatmap
    conv_layer = model.get_layer(layer_name)
    
    # Create DL model in reverse of given model
    heatmap_model = tf.keras.models.Model([model.inputs[0]], [conv_layer.output, model.output])

    # Pass through case through new model and get loss
    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(x)
        loss = predictions[0]

    # Compute gradients for output of new model w.r.t. loss
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Maximise gradients
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    # Generate weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Array for initial heatmap
    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    # Accumulate weights
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Resize and generate final heatmap
    cam = cv2.resize(cam.numpy(), (80*4, 60*4))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap

def get_mean_heatmap(dl_model: tf.keras.Model, save_name: str) -> None:

    '''
    Function to get and plot the mean heatmap from cases which have a positive inference
    args:
        dl_model: DL CNN-based model
        save_name: Name with which to save plot
    '''
    
    # Load data
    data, _, files = load_dl_data()
    
    # Get inferences
    results = dl_model.predict(data)
    
    # List for heatmaps
    heatmaps = []
    
    # For each result, check if inference is positive. If so, get heatmap and append to heatmaps list. Flip heatmap if case is from the Southern Hemisphere
    for res_i, res in enumerate(results):
        if res > 0.5:
            heatmap = get_heatmap(dl_model, "conv2d_1", data[res_i])
            if "-60" in files[res_i]:
                heatmaps.append(np.flipud(heatmap))
            else:
                heatmaps.append(heatmap)
    
    # Turn list of heatmaps into a numpy array
    heatmaps = np.array(heatmaps)
    
    # Generate lats
    lats = np.zeros(60*4)
    lats[0] = 0
    for i in range(1, len(lats)):
        lats[i] = lats[i-1] + 0.25

    # Generate lons
    lons = np.zeros(80*4)
    lons[0] = 0
    for i in range(1, len(lons)):
        lons[i] = lons[i-1] + 0.25
    
    # Plot mean heatmap
    plt.figure()
    plt.contourf(lons, lats, np.mean(heatmaps, axis=0))
    plt.colorbar()
    plt.savefig(save_name + ".pdf")
    
def get_base_DL_dict(dl_model: tf.keras.Model, from_file: bool) -> dict:

    '''
    Function to generate DL dict of TC centres without any constraints
    args:
        dl_model: Deep Learning model
        from_file: Generate from data (False) [slow] or from previously saved points (True) [quick]
    returns:
        tc_dict: dict of TC centres as lats/lons, with keys being date/time
    '''
    
    # Date bounds for dict
    start_date=dt(2017, 8, 1)
    end_date=dt(2019, 8, 31, 23, 59)

    # Set up empty dict
    tc_dict = {}
    curr_date = start_date
    while curr_date < end_date:
        tc_dict[curr_date] = []
        curr_date += td(hours=6)
    
    # If loading from DL cases (slow)
    if from_file == False:

        # Load data
        data, labels, files = load_dl_data()

        # Make dl predictions
        preds = dl_model.predict(data)

        # Filter out any data in which DL model infers on TC. Keep rest for heatmap calculation
        true_indices = []
        for pred_i, pred in enumerate(preds):
            if pred > 0.5:
                true_indices.append(pred_i)

        # Set up inputs for multiprocessing to generate lat/lon from heatmaps
        inputs = []
        
        # Loop through each of cases which have a positive inference
        for i, ind in enumerate(true_indices):
            
            # Get initial lat/lon from heatmap
            lat, lon = generate_initial_lat_lon_dl_case(data[ind], files[ind], dl_model)

            # Get name of file that generated the lat/lon
            name = files[ind].replace(".npz", "").split("/")[-1]
            
            # Get the date and time from the file name
            date_str = name.split("_")[-1]

            # Create a datetime object with the date and time
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(date_str[8:])
            date = dt(year, month, day, hour)

            if lon == 180:
                if "180_260" in files[ind]:
                    lon = 185.01
                else:
                    lon = 174.99
            elif lon == 100:
                if "100_180" in files[ind]:
                    lon = 105.01
                else:
                    lon = 94.99
            elif lon == 260:
                if "180_260" in files[ind]:
                    lon = 254.99
                else:
                    lon = 265.01
            
            # Reset lon into the range [-180, 180]
            if lon > 180:
                lon -= 360

            if lat == 0:
                if "-60_0" in files[ind]:
                    lat = -5.01
                else:
                    lat = 5.01
                
            # Append inputs to list
            inputs.append([date, lat, lon])
        
        # Open multiprocessing pool
        pool = mp.Pool(int(0.75*mp.cpu_count()))
        
        # Generate lat/lon for inputs from ERAI data and using mask centered on vorticity
        res = list(pool.map(generate_final_lat_lon_dl_case, inputs))
        
        # Close pool
        pool.close()

        # Loop through each date/lat/lon case generated and add to tc_dict
        for r in res:
            date, lat, lon = r
            
            # Reset lon into the range [-180, 180]
            if lon > 180:
                lon -= 360

            tc_dict[date].append([lat, lon])
        
    else: # Quick
        
        # Generate tc_dict from text file
        with open("dl_dict.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.replace("\n", "").split(";")
                date_time_str = split_line[0]
                tcs = split_line[1:-1]
                date, time = date_time_str.split(" ")
                year, month, day = map(int, date.split("-"))
                date_time = dt(year, month, day, int(time[:2]))
                
                if date_time in tc_dict.keys():
                    for tc in tcs:
                        lat, lon = map(float, tc.split(","))
                        if lon > 180:
                            lon -= 360
                        tc_dict[date_time].append([lat, lon])
                    
    return tc_dict

def generate_initial_lat_lon_dl_case(data: np.array, file: str, dl_model: tf.keras.Model) -> tuple(float, float):

    '''
    Function that generates a lat/lon pair from a case of data previously generated from ERAI and preprocessed via its CAM heatmap
    args:
        data: numpy array of data
        file: file pathfrom which the data was loaded
        dl_model: Deep Learning CNN-based model from which to generate a heatmap fo the case given
    returns:
        lat: latitude for point where the maximum value in the heatmap is
        lat: longitude for point where the maximum value in the heatmap is
    '''
    
    # Get name of file
    name = file.replace(".npz", "").split("/")[-1]
    
    # Get region bounding box from file name
    min_lat, max_lat, min_lon, max_lon = map(int, name.split("_")[-5:-1])
    
    # Get heatmap
    heatmap = get_heatmap(dl_model, "conv2d_1", data) 
    
    # Generate lats for case
    lats = np.zeros(60*4)
    lats[0] = min_lat
    for i in range(1, len(lats)):
        lats[i] = lats[i-1] + 0.25

    # Generate lons for case
    lons = np.zeros(80*4)
    lons[0] = min_lon
    for i in range(1, len(lons)):
        lons[i] = lons[i-1] + 0.25
    
    # Get indices for lat/lon from max in heatmap
    max_row, max_col = np.unravel_index(heatmap.argmax(), heatmap.shape)
    
    # Transform lat/lon indices to actual lat/lon
    lon = lons[max_col]
    lat = lats[max_row]
    
    return lat, lon

def generate_final_lat_lon_dl_case(inputs: list[dt, float, float]) -> list[dt, float, float]:

    '''
    Function to get a better lock on a TC center from that given via a heatmap. Centers are generated to be at max vorticity in a 10deg region around initial TC center.
    args:
        inputs:
            date: date of case
            lat: latitude of TC center as given by heatmap
            lon: longitude of TC center as given by heatmap
    returns:
        date: date of case
        lat: new latitude of TC center
        lon: new longitude of TC center
    '''
    
    # Get inputs
    date, lat, lon = inputs
    
    # Reset lat/lon to make sure new lat/lon stay in same section
    if 0 < (100 - lon) < 5:
        lon = 94.5
    elif 0 < (lon - 100) < 5:
        lon = 105.5
    elif 0 < (180 - lon) < 5:
        lon = 174.5
    elif 0 < (180 + lon) < 5:
        lon = -174.5
    elif 0 < (100 + lon) < 5:
        lon = -94.5
    elif 0 < (-lon - 100) < 5:
        lon = -105.5
    elif 0 < (-lon - 20) < 5:
        lon = -25.5
    elif 20 <= lon <= 25:
        lon = 25.5
    
    if 0 < (60 - lat) < 5:
        lat = 54.5
    elif 0 < lat < 5:
        lat = 5.5
    elif -5 < lat < 0:
        lat = -5.5
    elif 0 < (60 + lat) < 5:
        lat = -54.5
    
    # Get data from ERAI
    nc_data, lats, lons = load_nc_file([date, lat, lon, 10])
    
    # Isolate vorticity fields
    vort850 = nc_data[:, :, 2]
    vort700 = nc_data[:, :, 3]
    vort600 = nc_data[:, :, 4]

    # Scale each vorticity field using each height's min value. This is so mask is not overinfluenced by lower height.
    vort850 = 2.*(vort850 - np.min(vort850))/np.ptp(vort850)-1
    vort700 = 2.*(vort700 - np.min(vort700))/np.ptp(vort700)-1
    vort600 = 2.*(vort600 - np.min(vort600))/np.ptp(vort600)-1
    
    # Create mask of vorticity by getting the mean vorticity
    mask = np.mean([vort850, vort700, vort600], axis=0)
    
    # Get indices for point of max mean vorticity
    max_row, max_col = np.unravel_index(mask.argmax(), mask.shape)
        
    # Generate lat/lon from indices
    new_lon = lons[max_col]
    new_lat = lats[max_row]
    
    if new_lon > 180:
        new_lon -= 360
    
    return date, new_lat, new_lon

def get_IBTRACS_dict(ibtracs_path: str = "ibtracs.ALL.list.v04r00.csv", start_date: dt = dt(2017, 8, 1), end_date: dt = dt(2019, 8, 31, 23, 59), cat_opt: int = 0, min_cat: int = 0) -> dict:

    '''
    Parse IBTrACS CSV to get a dict of TCs
    args:
        ibtracs_path: path to IBTrACS csv
        start_date: date to start processing from
        end_date: date to end processing at
        cat_opt: select output type for each TC point (0 = [lat, lon], 1 = [lat, lon, catagory]
        min_cat: minimum category to consider in [-7, 5]
    returns:
        storms: dict of storms that were present in the timestep
    '''
    
    # Set up empty dict
    tc_dict = {}

    # Create keys
    curr_date = start_date
    while curr_date < end_date:
        tc_dict[curr_date] = []
        curr_date += td(hours=6)

    # List of tracks
    tracks = []     

    with open(ibtracs_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        track = []
        old_sid = None
        old_name = None
        
        # For each row in the CSV file
        for i, row in enumerate(readCSV):

            # Get rid of headers
            if i < 2:
                continue

            sid = row[0]
            name = row[5]
                
            # Get date/time
            date, time = row[6].split(" ")
            year, month, day = map(int, date.split("-"))
            hour = int(time.split(":")[0])
            date_time = dt(year, month, day, hour)

            # Do not consider if date/time not in test region
            if date_time < start_date or date_time > end_date or hour%6 != 0:
                continue

            # Get storm data
            storm_type = row[7]
            lat = row[8]
            lon = row[9]
            
            wmo_wind = row[10]
            wmo_press = row[11]
            usa_wind = row[23]
            usa_press = row[24]
            tokyo_wind = row[45]
            tokyo_press = row[46]
            cma_wind = row[57]
            cma_press = row[58]
            hko_wind = row[62]
            hko_press = row[63]
            newdelhi_wind = row[67]
            newdelhi_press = row[68]
            reunion_wind = row[75]
            reunion_press = row[76]
            bom_wind = row[95]
            bom_press = row[96]
            nadi_wind = row[120]
            nadi_press = row[121]
            wellington_wind = row[124]
            wellington_press = row[125]
            ds824_wind = row[129]
            ds824_press = row[130]
            td9636_wind = row[134]
            td9636_press = row[135]
            neumann_wind = row[144]
            neumann_press = row[145]
            mlc_wind = row[149]
            mlc_press = row[150]
            usa_cat = row[25] 

            # get wind value
            wind = -1
            wind_choice = [wmo_wind, usa_wind, tokyo_wind, cma_wind, hko_wind, newdelhi_wind, reunion_wind, bom_wind, nadi_wind, wellington_wind, ds824_wind, td9636_wind, neumann_wind, mlc_wind]
            for choice in wind_choice:
                if choice.strip() and wind == -1 :
                    wind = int(choice)
                    break

            # get pressure value
            press = -1
            press_choice = [wmo_press, usa_press, tokyo_press, cma_press, hko_press, newdelhi_press, reunion_press, bom_press, nadi_press, wellington_press, ds824_press, td9636_press, neumann_press, mlc_press]
            for choice in press_choice:
                if choice.strip() and press == -1:
                    press =	int(choice)
                    break

            # get category of TC
            try:
                usa_cat = int(usa_cat)
            except:
                if storm_type[0] == "E": #post-tropical
                    usa_cat = -4
                elif storm_type in ["WV", "LO", "DB", "DS", "IN", "MD"]: #misc disturbances
                    usa_cat = -3
                elif storm_type[0] == "SS": #subtropical
                    usa_cat = -2
                if wind < 34 and storm_type[0] == "T": # tropical depression
                    usa_cat = -1
                elif 34 <= wind < 64 and storm_type[0] == "T": # tropical storm
                    usa_cat = 0
                elif 64 <= wind < 83: #cat 1
                    usa_cat = 1
                elif 83 <= wind < 96: #cat 2
                    usa_cat = 2
                elif 96 <= wind < 113: #cat 3
                    usa_cat = 3
                elif 113 <= wind < 137: #cat 4
                    usa_cat = 4
                elif wind >= 137: #cat 5
                    usa_cat = 5 
                else:
                    usa_cat = -5

            #If TC
            if usa_cat > min_cat:
                
                # Reset lon to be in region [-180, 180]
                lon = float(lon)
                if lon > 180:
                    lon -= 360
                
                # if new track, store old one and create new one
                if name != old_name or sid != old_sid:
                    tracks.append(copy.deepcopy(track))
                    track = []
                    old_name = name
                    old_sid = sid
                    
                # add TC data to track
                track.append([date_time, float(lat), lon])
                if cat_opt == 1:
                    tc_dict[date_time].append([float(lat), lon, usa_cat])
                else:
                    tc_dict[date_time].append([float(lat), lon])

        # add track to tracks list if one remains at last timestep         
        if len(track) > 0:
            tracks.append(copy.deepcopy(track))

    # remove tracks with length == 0   
    good_tracks = []
    for track in tracks:
        if len(track) > 0:
            good_tracks.append(track)
    tracks = good_tracks
                
    #return dict whose keys are dates and each key has a list of storms that happened on the date/time
    #each storm is listed as [cat, lat, lon, max wind, min MSLP]
    return tc_dict, tracks


def get_TRACK_tracks(paths: list[str]) -> tuple(dict, list[list[dt, float, float]]):

    '''
    Obtain tracks from TRACK-like outputs
    args:
        paths: list of paths which point to TRACK-like outputs
    returns:
        tc_dict: dict of TC centres as lats/lons, with keys being date/time
        tracks: list of TC tracks
    '''
    
    # Date bounds
    start_date=dt(2017, 8, 1)
    end_date=dt(2019, 8, 31, 23, 59)
    
    # Set up empty dict
    tc_dict = {}
    curr_date = start_date
    while curr_date < end_date:
        tc_dict[curr_date] = []
        curr_date += td(hours=6)

    # Set up list of tracks
    tracks = []
    
    # For each TRACK file
    for path_i, path in enumerate(paths):
        with open(path, "r") as file:
            
            # Read all lines in file
            lines = file.readlines()
            
            # Set up list of TC centres for current track
            track = []
            for line in lines:
                line = line.split(" & ")
                
                # If line does not have a TC centre, add current track to list of tracks and start a new track
                if len(line)<5:
                    if len(track) > 0:
                        tracks.append(copy.deepcopy(track))
                        track = []
                    continue
                    
                # Get TC centre
                date_time_str, lon, lat, _ = line[0].split(" ")
                lon = float(lon)
                lat = float(lat)
                year = int(date_time_str[0:4])
                month = int(date_time_str[4:6])
                day = int(date_time_str[6:8])
                hour = int(date_time_str[8:])
                date_time = dt(year, month, day, hour)
                
                if lon > 180:
                    lon -= 360
                    
                # If TC centre within the date bounds, add it to current track and dict
                if start_date < date_time < end_date:
                    tc_dict[date_time].append([lat, lon])
                    track.append([date_time, lat, lon])
            
            # If there is a viable track at the end of the file, add it to list of tracks
            if len(track) > 0:
                tracks.append(copy.deepcopy(track))
          
    # pass over tracks to remove any empty tracks
    good_tracks = []
    for track in tracks:
        if len(track) > 0:
            good_tracks.append(track)
    tracks = good_tracks
        
    return tc_dict, tracks
    
def get_DL_dict(dl_model: tf.keras.Model, from_file: bool = True, long_tracks: bool = False) -> tuple(dict, list[list[dt, float, float]]):
    
    '''
    Obtain dict of TC centres as given by TCDetect
    args:
        dl_model: Loaded TCDetect
        from_file: obtain from previously saved results
        long_tracks: pass initial tracks to join short tracks together 
    returns:
        tc_dict: dict of TC centres as lats/lons, with keys being date/time
        tracks: list of TC tracks
    '''
    
    # Get TC dict
    tc_dict = get_base_DL_dict(dl_model, from_file)
    
    # basins
    lims = []
    lims.append([0, 60, 20, 100])
    lims.append([0, 60, 100, 180])
    lims.append([0, 60, -180, -100])
    lims.append([0, 60, -100, -20])
    lims.append([-60, 0, 20, 100])
    lims.append([-60, 0, 100, 180])
    lims.append([-60, 0, -180, -100])
    lims.append([-60, 0, -100, -20])
    
    # Set up list of tracks
    tracks = []
    
    # for each region
    for region in lims:
        
        # get region bounds
        min_lat, max_lat, min_lon, max_lon = region
        
        # get all TCs detected by TCDetect in region over all timesteps
        sec_dict = {}
        for key in tc_dict.keys():
            sec_dict[key] = []
            sec_storms = get_storms_in_section(tc_dict[key], min_lat, max_lat, min_lon, max_lon)
            
            for sec_storm in sec_storms:
                sec_dict[key].append(sec_storm)
        
        # join TC centres into tracks
        track = []
        for key in sec_dict.keys():
            if len(sec_dict[key]) == 0:
                if len(track) > 0:
                    tracks.append(copy.deepcopy(track))
                    track = []
            else:
                track.append([key, sec_dict[key][0][0], sec_dict[key][0][1]])
    
        # if last track is not empty, add it to list of tracks
        if len(track) > 0:
            tracks.append(copy.deepcopy(track))
        
    # Remove any empty tracks
    good_tracks = []
    for track in tracks:
        if len(track) > 0:
            good_tracks.append(track)
    tracks = good_tracks
    
    # If joining short tracks together
    if long_tracks == True:
        
        while True:

            # set up list of new tracks
            new_tracks = []
            
            # set flag and counter to know when to exit
            continue_flag = 0
            counter = 0
            
            # for each original track
            for track_i in range(len(tracks)-1):

                # check if time to exit
                if continue_flag == 1:
                    continue_flag = 0
                    continue

                # get last point of current track
                end_track_1 = tracks[track_i][-1]
                
                # get first point of next track
                start_track_2 = tracks[track_i + 1][0]

                # get dates of the two points
                dates = [end_track_1[0], start_track_2[0]]
                dates.sort()

                # calculate time difference between two points
                diff = dates[1] - dates[0]
                days = diff.days
                hours = diff.seconds/(60*60.)
                days += hours/24

                # if time diff between the two points <= 2 days and distance <= 20 deg (geodesic), combine the two tracks
                if days <= 2 and geodesic_distance(end_track_1[2], end_track_1[1], start_track_2[2], start_track_2[1]) <= 20:

                    new_track = tracks[track_i] + tracks[track_i + 1]
                    new_tracks.append(new_track)
                    
                    # log that a change was made
                    continue_flag = 1
                    counter += 1
                    
                # else add each to track list
                else:

                    new_tracks.append(tracks[track_i])
                    if track_i == len(tracks) - 1:
                        new_tracks.append(tracks[track_i+1])

            # replace track list for next pass
            tracks = new_tracks

            # if no changes made to tracks, exit
            if counter == 0:
                break

    return tc_dict, tracks

def get_storms_in_section(storms: list[list[dt, float, float]], min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> list[list[dt, float, float]]:

    '''
    Function to get storms in a region from a list of storms
    args:
        storms: list of TC centres in lat/lon pairs
        min_lat, max_lat, min_lon, max_lon: bounding coordinates for region to find TCs in
    returns:
        sec_storms: list of TC centres inside region
    '''
    
    # Initialise list for TC centres in region
    sec_storms = []
    
    # Loop through list of TCs
    for storm in storms:
        
        # Get lat, lon of TC
        if len(storm) == 5:
            _, lat, lon, _, _ = storm
        else:
            lat, lon = storm
        
        # If TC is in region, add it to list
        if min_lat <= lat < max_lat and min_lon <= lon < max_lon:
            sec_storms.append([lat, lon])
            
    return sec_storms

def geodesic_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    '''
    Function to get the geodesic distance between two lat/lon pairs
    args:
        lon1, lat1: lon/lat pair for first point
        lon2, lat2: lon/lat pair for second point
    returns:
        dist: geodesic distance between teo points on the globe
    '''
    
    # get geodesic distance
    dist = geodesic((lat1, lon1), (lat2, lon2)).km
    
    # turn it into radians
    dist_deg = (dist / 6371.) * (180. / math.pi)
    
    return dist_deg

def generate_venn_numbers(ib_dict: dict, tr_dict: dict, dl_dict: dict, hemi: int) -> tuple(int, int, int, int, int, int, int):

    '''
    Function to generate Venn diagram section numbers.
    args:
        ib_dict: dict of TC centres as cenerated by IBTrACS with keys being datetime objects
        tr_dict: dict of TC centres as cenerated by T-/TRACK with keys being datetime objects
        dl_dict: dict of TC centres as cenerated by the Deep Learning model with keys being datetime objects
        hemi: hemisphere for which to process TCs; 0 = all, 1 = NH, 2 = SH
    returns:
        dl: updated number of regions processed that fall in the DL section of the Venn diagram
        ib: updated number of regions processed that fall in the IBTrACS section of the Venn diagram
        tr: updated number of regions processed that fall in the T-/TRACK section of the Venn diagram
        dl_ib: updated number of regions processed that fall in the DL and IBTrACS section of the Venn diagram
        dl_tr: updated number of regions processed that fall in the DL and T-/TRACK section of the Venn diagram
        ib_tr: updated number of regions processed that fall in the IBTrACS and T-/TRACK section of the Venn diagram
        all_methods: updated number of regions processed that fall in the All Methods section of the Venn diagram      
    '''
    
    # Set up initial and final dates to process TCs for
    start_date = dt(2017, 8, 1)
    end_date = dt(2019, 8, 31, 23, 59)
    
    # Initialise Venn diagram section numbers
    dl = 0
    ib = 0
    tr = 0
    dl_ib = 0
    dl_tr = 0
    ib_tr = 0
    all_methods = 0
    
    # Initialise list for TC lists
    tr_points = []
    ib_points = []
    dl_points = []
    tr_ib_points = []
    tr_dl_points = []
    ib_dl_points = []
    all_methods_points = []
        
    # Loop through all dates
    curr_date = start_date
    while curr_date < end_date:
        
        # If date/time not included in at least one of the dicts, skip over date
        if curr_date not in ib_dict.keys() or curr_date not in tr_dict.keys() or curr_date not in dl_dict.keys():
            
            curr_date += td(hours = 6)
            continue
        
        # Get TCs from IBTrACS for current date
        ib_storms = ib_dict[curr_date]
        
        # Get TCs from T-/TRACK for current date
        tr_storms = tr_dict[curr_date]
        
        # Get TCs from DL for current date
        dl_storms = dl_dict[curr_date]
        
        # If Northern Hemisphere wanted
        if hemi == 0 or hemi == 1:
            
            # Get Venn section type for North Indian region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, 0, 60, 20, 100)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for North Western Pacific region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, 0, 60, 100, 180)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for North Eastern Pacific region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, 0, 60, -180, -100)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for North Atlantic region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, 0, 60, -100, -20)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
        # If Southern Hemisphere wanted
        if hemi == 0 or hemi == 2:
            
            # Get Venn section type for Southern Indian region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, -60, 0, 20, 100)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for South Eastern Pacific region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, -60, 0, 100, 180)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for Souther Western Pacific region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, -60, 0, -180, -100)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
            
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
            # Get Venn section type for South Atlantic region
            venn_type, new_points = get_venn_type_coords_for_venn_section(ib_storms, tr_storms, dl_storms, -60, 0, -100, -20)
            
            # Update Venn diagram section numbers
            dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods = update_venn_numbers(venn_type, dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods)
        
            # Update TC centre list
            tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = update_points(tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points, new_points, venn_type, curr_date)
            
        # Move on to next date
        curr_date += td(hours = 6)
        
    return dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods, tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points

def get_venn_type_coords_for_venn_section(ib_storms: list[list[dt, float, float]], tr_storms: list[list[dt, float, float]], dl_storms: list[list[dt, float, float]], min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> tuple(int, list[list[dt, float, float]]):

    '''
    Function to check to which Venn section a given region belongs, and the respective TC centres
    args:
        ib_storms: list of TC centres (lat/lon pairs) comnig from IBTrACS
        tr_storms: list of TC centres (lat/lon pairs) comnig from T-/TRACK
        dl_storms: list of TC centres (lat/lon pairs) comnig from DL
        min_lat, max_lat, min_lon, max_lon: coordinates for region bounding box
    returns:
        venn_type: Venn section number [-1: None; 0: IBTrACS; 1: T-/TRACK; 2: DL; 3: IBTrACS and T-/TRACK; 4: T-/TRACK and DL; 5: IBTrACS and DL; 6: All Methods]
        tc_centres: list of all TCs in the region; source is T-/TRACK if it is component of region, IBTrACS if T-/TRACK not present, DL if only DL TCs present
    '''
    
    # Get TC centres in region from IBTrACS
    sec_ib_storms = get_storms_in_section(ib_storms, min_lat, max_lat, min_lon, max_lon)            
    
    # Get TC centres in region from T-/TRACK
    sec_tr_storms = get_storms_in_section(tr_storms, min_lat, max_lat, min_lon, max_lon)
    
    # Get TC centres in region from DL
    sec_dl_storms = get_storms_in_section(dl_storms, min_lat, max_lat, min_lon, max_lon)

    # Return Venn type and list of TCs
    if len(sec_ib_storms) == 0 and len(sec_tr_storms) == 0 and len(sec_dl_storms) == 0:
        return -1, None
    elif len(sec_ib_storms) > 0 and len(sec_tr_storms) == 0 and len(sec_dl_storms) == 0:
        return 0, sec_ib_storms
    elif len(sec_ib_storms) == 0 and len(sec_tr_storms) > 0 and len(sec_dl_storms) == 0:
        return 1, sec_tr_storms
    elif len(sec_ib_storms) == 0 and len(sec_tr_storms) == 0 and len(sec_dl_storms) > 0:
        return 2, sec_dl_storms
    elif len(sec_ib_storms) > 0 and len(sec_tr_storms) > 0 and len(sec_dl_storms) == 0:
        return 3, sec_tr_storms
    elif len(sec_ib_storms) == 0 and len(sec_tr_storms) > 0 and len(sec_dl_storms) > 0:
        return 4, sec_tr_storms
    elif len(sec_ib_storms) > 0 and len(sec_tr_storms) == 0 and len(sec_dl_storms) > 0:
        return 5, sec_ib_storms
    elif len(sec_ib_storms) > 0 and len(sec_tr_storms) > 0 and len(sec_dl_storms) > 0:
        return 6, sec_tr_storms
    else:
        print("Error")
        return None

def update_venn_numbers(venn_type: int, dl: int, ib: int, tr: int, dl_ib: int, dl_tr: int, ib_tr: int, all_methods: int) -> tuple(int, int, int, int, int, int, int):

    '''
    Function that updates the Venn diagram section numbers
    args:
        venn_type: type of region to add to
        dl: number of regions processed that fall in the DL section of the Venn diagram
        ib: number of regions processed that fall in the IBTrACS section of the Venn diagram
        tr: number of regions processed that fall in the T-/TRACK section of the Venn diagram
        dl_ib: number of regions processed that fall in the DL and IBTrACS section of the Venn diagram
        dl_tr: number of regions processed that fall in the DL and T-/TRACK section of the Venn diagram
        ib_tr: number of regions processed that fall in the IBTrACS and T-/TRACK section of the Venn diagram
        all_methods: number of regions processed that fall in the All Methods section of the Venn diagram
    returns:
        dl: updated number of regions processed that fall in the DL section of the Venn diagram
        ib: updated number of regions processed that fall in the IBTrACS section of the Venn diagram
        tr: updated number of regions processed that fall in the T-/TRACK section of the Venn diagram
        dl_ib: updated number of regions processed that fall in the DL and IBTrACS section of the Venn diagram
        dl_tr: updated number of regions processed that fall in the DL and T-/TRACK section of the Venn diagram
        ib_tr: updated number of regions processed that fall in the IBTrACS and T-/TRACK section of the Venn diagram
        all_methods: updated number of regions processed that fall in the All Methods section of the Venn diagram        
    '''
    
    # Update numbers according to case type
    if venn_type == -1:
        pass
    elif venn_type == 0:
        ib += 1
    elif venn_type == 1:
        tr += 1
    elif venn_type == 2:
        dl += 1
    elif venn_type == 3:
        ib_tr += 1
    elif venn_type == 4:
        dl_tr += 1
    elif venn_type == 5:
        dl_ib += 1
    elif venn_type == 6:
        all_methods += 1
    else:
        print("Error", venn_type)
    
    return dl, ib, tr, dl_ib, dl_tr, ib_tr, all_methods
    
def update_points(tr_points: list[list[dt, float, float]], ib_points: list[list[dt, float, float]], dl_points: list[list[dt, float, float]], tr_ib_points: list[list[dt, float, float]], tr_dl_points: list[list[dt, float, float]], ib_dl_points: list[list[dt, float, float]], all_methods_points: list[list[dt, float, float]], new_points: list[float, float], venn_sec: int, date: dt):

    '''
    Function that updates the TC centres in each Venn diagram section
    args:
        tr_points: list of TC centres in the T-/TRACK only Venn section at a given date
        ib_points: list of TC centres in the IBTrACS only Venn section at a given date
        dl_points: list of TC centres in the DL only Venn section at a given date
        tr_ib_points: list of TC centres in the T-/TRACK and IBTrACS Venn section at a given date
        tr_dl_points: list of TC centres in the T-/TRACK and DL Venn section at a given date
        ib_dl_points: list of TC centres in the IBTrACS and DL Venn section at a given date
        all_methods_points: list of TC centres in the All Methods Venn section at a given date
        new_points: lost of TC centres to be added
        venn_sec: Index for Venn section to which to add new_points
        date: date at which the TCs in new_points occurred
    returns:
        tr_points: list of TC centres in the T-/TRACK only Venn section
        ib_points: list of TC centres in the IBTrACS only Venn section
        dl_points: list of TC centres in the DL only Venn section
        tr_ib_points: list of TC centres in the T-/TRACK and IBTrACS Venn section
        tr_dl_points: list of TC centres in the T-/TRACK and DL Venn section
        ib_dl_points: list of TC centres in the IBTrACS and DL Venn section
        all_methods_points: list of TC centres in the All Methods Venn section     
    '''
    
    # Add TC centre to relevant section
    if venn_sec == -1: # If no TCs are present
        pass
    elif venn_sec == 0: # Add TCs to DL only Venn section
        for pt in new_points:
            lat, lon = pt
            ib_points.append([date, lat, lon])
    elif venn_sec == 1: # Add TCs to DL only Venn section
        for pt in new_points:
            lat, lon = pt
            tr_points.append([date, lat, lon])
    elif venn_sec == 2: # Add TCs to DL only Venn section
        for pt in new_points:
            lat, lon = pt
            dl_points.append([date, lat, lon])
    elif venn_sec == 3: # Add TCs to T-/TRACK and IBTrACS Venn section
        for pt in new_points:
            lat, lon = pt
            tr_ib_points.append([date, lat, lon])
    elif venn_sec == 4: # Add TCs to T-/TRACK and DL Venn section
        for pt in new_points:
            lat, lon = pt
            tr_dl_points.append([date, lat, lon])
    elif venn_sec == 5: # Add TCs to IBTrACS and DL Venn section
        for pt in new_points:
            lat, lon = pt
            ib_dl_points.append([date, lat, lon])
    elif venn_sec == 6: # Add TCs to All Methods Venn section
        for pt in new_points:
            lat, lon = pt
            all_methods_points.append([date, lat, lon])
    else:
        print("Error in point generation")

    return tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points

def get_section_id(pt: list[float, float]) -> int:

    '''
    Obtain section id given lat/lon
    args:
        pt: list of lat and lon
    returns:
        section_id: ID of section in which point is in
    '''
    
    # Get lat/lon
    lat, lon = pt
    
    # Return section id
    if lat > 0:
        if 20 < lon <= 100:
            return 0
        elif 100 < lon <= 180:
            return 1
        elif -180 < lon <= -100:
            return 2
        elif -100 < lon <= -20:
            return 3
    else:
        if 20 < lon <= 100:
            return 4
        elif 100 < lon <= 180:
            return 5
        elif -180 < lon <= -100:
            return 6
        elif -100 < lon <= -20:
            return 7
    return -1

def get_storms_in_section_by_id(storms: list, section_id: int) -> list[list]:

    '''
    Function to get storms in a region from a list of storms
    args:
        storms: list of TC centres in lat/lon pairs
        min_lat, max_lat, min_lon, max_lon: bounding coordinates for region to find TCs in
    returns:
        sec_storms: list of TC centres inside region
    '''
    
    # Get section bounds from it's id
    if section_id == 0:
        min_lon, max_lon, min_lat, max_lat = [20, 100, 0, 60]
    elif section_id == 1:
        min_lon, max_lon, min_lat, max_lat = [100, 180, 0, 60]
    elif section_id == 2:
        min_lon, max_lon, min_lat, max_lat = [-180, -100, 0, 60]
    elif section_id == 3:
        min_lon, max_lon, min_lat, max_lat = [-100, -20, 0, 60]
    elif section_id == 4:
        min_lon, max_lon, min_lat, max_lat = [20, 100, -60, 0]
    elif section_id == 5:
        min_lon, max_lon, min_lat, max_lat = [100, 180, -60, 0]
    elif section_id == 6:
        min_lon, max_lon, min_lat, max_lat = [-180, -100, -60, 0]
    elif section_id == 7:
        min_lon, max_lon, min_lat, max_lat = [-100, -20, -60, 0]
    else:
        return []
    
    # Initialise list for TC centres in region
    sec_storms = []
    
    # Loop through list of TCs
    for storm in storms:
        
        # Get lat, lon of TC
        if len(storm) == 5:
            _, lat, lon, _, _ = storm
        elif len(storm) == 3:
            lat, lon, cat = storm
        else:
            lat, lon = storm
        
        # If TC is in region, add it to list
        if min_lat <= lat < max_lat and min_lon <= lon < max_lon:
            if len(storm) >= 3:
                sec_storms.append([lat, lon, cat])
            else:
                sec_storms.append([lat, lon])    
            
    return sec_storms

def dl_overlapping_tracks(s1_tracks: list[list[dt, float, float]], dl_dict: list[list[dt, float, float]], dl_tracks: list[list[dt, float, float]]) -> tuple(list[list[dt, float, float]], list[list[dt, float, float]], list[list[dt, float, float]], list[list[dt, float, float]]):

    '''
    Get overlapping tracks vs DL tracks
    args:
        s1_tracks: list of tracks from IBTrACS or T-/TRACK
        dl_dict: TC dict from TCDetect
        dl_tracks: list of tracks from TCDetect
    returns:
        s1_only: tracks from s1 that do not match with any in dl
        matching_s1: s1 version of tracks that match with dl
        matching_dl: dl version of tracks that match with s1
        dl_only: tracks from dl that do not match with any in s1
    '''
    
    # Set up lists
    s1_only = []
    matching_s1 = []
    matching_dl = []
    
    # for each base track
    for s1_i, s1_track in enumerate(s1_tracks):
        
        break_flag = False
        
        # for each TC in base track
        for pt in s1_track:
            
            # get TC points
            date, lat, lon = pt
            
            # get any TCs from DL in the same section and timestep
            dl_storms = get_storms_in_section_by_id(dl_dict[date], get_section_id([lat, lon]))
            
            # if there are TCs present
            if len(dl_storms) > 0:                
                
                # find track having the TC point and add tracks to relevant list
                for dl_i, dl_track in enumerate(dl_tracks):
                    dl_break_flag = False
                    for dl_pt in dl_track:
                        dl_date, dl_lat, dl_lon = dl_pt
                        if dl_date == date and dl_lat == dl_storms[0][0] and dl_lon == dl_storms[0][1]:
                            matching_s1.append(s1_i)
                            matching_dl.append(dl_i)
                            dl_break_flag = True
                            break
                    if dl_break_flag == True:
                        break
                        
                break_flag = True
                break
        
        # if no matching track, add it to list
        if break_flag == True:
            break_flag = False
        else:
            s1_only.append(s1_i)
            
    # get list of DL only tracks
    dl_only = list(set(range(len(dl_tracks))) - set(matching_dl))
    
    return s1_only, matching_s1, matching_dl, dl_only

def overlapping_tracks(t_track_tracks: list[list[dt, float, float]], dl_dict: list[list[dt, float, float]], dl_tracks: list[list[dt, float, float]], ibtracs_tracks: list[list[dt, float, float]]):

    '''
    Get overlapping tracks across the three sources
    args:
        t_track_tracks: list of tracks from T-/TRACK
        dl_dict: TC dict from TCDetect
        dl_tracks: list of tracks from TCDetect
        ibtracs_tracks: list of tracks from IBTrACS
    '''
    
    # get matching tracks for IBTrACS vs DL
    ib_only, matching_ib_dl_ib, matching_ib_dl_dl, dl_only_ib = dl_overlapping_tracks(ibtracs_tracks, dl_dict, dl_tracks)
    
    # get matching tracks for T-/TRACK vs DL
    tr_only, matching_tr_dl_tr, matching_tr_dl_dl, dl_only_tr = dl_overlapping_tracks(t_track_tracks, dl_dict, dl_tracks)
    
    # DL only tracks
    dl_only = list(set(dl_only_ib + dl_only_tr))
    
    # get DL versions of tracks that matched with all 3 sources
    all_dl = []
    for i in range(len(matching_ib_dl_dl)):
        if matching_ib_dl_dl[i] in matching_tr_dl_dl:
            all_dl.append(matching_ib_dl_dl[i])
    for i in range(len(matching_tr_dl_dl)):
        if matching_tr_dl_dl[i] in matching_ib_dl_dl:
            all_dl.append(matching_tr_dl_dl[i])
    
    # get IBTrACS versions of tracks that matched with all 3 sources
    all_ib = []
    for i in range(len(matching_ib_dl_dl)):
        if matching_ib_dl_dl[i] in all_dl:
            all_ib.append(matching_ib_dl_ib[i])
    
    # get T-/TRACK versions of tracks that matched with all 3 sources
    all_tr = []
    for i in range(len(matching_ib_dl_dl)):
        if matching_tr_dl_dl[i] in all_dl:
            all_tr.append(matching_tr_dl_tr[i])
    
    # get T-/TRACK and IBTrACS versions of tracks that matched
    tr_ib_tr = []
    tr_ib_ib = []
    for ib_i in ib_only:
        for ib_pt in ibtracs_tracks[ib_i]:
            ib_date, ib_lat, ib_lon = ib_pt
            ib_sec_id = get_section_id([ib_lat, ib_lon])
            
            break_flag = False
            for tr_i in tr_only:
                for tr_pt in t_track_tracks[tr_i]:
                    tr_date, tr_lat, tr_lon = tr_pt
                    tr_sec_id = get_section_id([tr_lat, tr_lon])
                    if ib_date == tr_date and tr_sec_id == ib_sec_id:
                        tr_ib_tr.append(tr_i)
                        tr_ib_ib.append(ib_i)
                        break_flag = True
                        break
                if break_flag == True:
                    break
            if break_flag == True:
                break            

    # print results
    print("IB:", len(ib_only)-len(tr_ib_tr))
    print("DL:", len(dl_only))
    print("TR:", len(tr_only)-len(tr_ib_tr))
    print("IB-DL:", len(matching_ib_dl_ib) - len(all_ib))
    print("DL-TR:", len(matching_tr_dl_tr) - len(all_tr))
    print("TR-IB:", len(tr_ib_tr))
    print("ALL:", len(all_ib))
    
def get_correlations(ibtracs_tracks: list[list[dt, float, float]], track_tracks: list[list[dt, float, float]], dl_tracks: list[list[dt, float, float]]) -> tuple(list[int], list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float],  list[float]):
    
    '''
    Function to calculate the differences between tracks from different sources
    args:
        ibtracs_tracks: list of tracks from IBTrACS
        track_tracks: list of track from T-/TRACK
        dl_tracks: list of tracks from DL
    returns:
        venn_sections: list of number of diffs in each section
        diff_x_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for points at which only IBTrACS and DL tracks overlap
        diff_y_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for points at which only IBTrACS and DL tracks overlap
        diff_x_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_ib_track: list of differences in latgitude, i.e. lat_ib - lat_track, for oints at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_track_dl: list of differences in longitude, i.e. lon_track - lon_dl, for points at which only T-/TRACK and DL tracks overlap
        diff_y_track_dl: list of differences in latitude, i.e. lat_track - lat_dl, for points at which only T-/TRACK and DL tracks overlap
        diff_x_all_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for points at which only IBTrACS and DL tracks overlap
        diff_y_all_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for points at which only IBTrACS and DL tracks overlap
        diff_x_all_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_all_ib_track: list of differences in latitude, i.e. lat_ib - lat_dl, for points at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_all_track_dl: list of differences in longitude, i.e. lon_tr - lon_dl, for points at which all sources overlap
        diff_y_all_track_dl: list of differences in latitude, i.e. lat_tr - lat_dl, for points at which all sources overlap
    '''
    
    #Calculate the percentage overlap and mean separation distance for tracks that overlap between IBTrACS tracks and T-/TRACK tracks
    perc_overlap_matrix_ibtracs_track = np.zeros((len(ibtracs_tracks), len(track_tracks)))
    dist_matrix_ibtracs_track = np.zeros((len(ibtracs_tracks), len(track_tracks)))
    dist_matrix_ibtracs_track[:, :] = 1e6
    
    for ib_track_i, ib_track in enumerate(ibtracs_tracks):
        for track_track_i, track_track in enumerate(track_tracks):
            mean_dist, perc_overlap = get_dist_between_tracks(ib_track, track_track)
            if perc_overlap > 0:
                dist_matrix_ibtracs_track[ib_track_i, track_track_i] = mean_dist
                perc_overlap_matrix_ibtracs_track[ib_track_i, track_track_i] = perc_overlap

    #Calculate the percentage overlap and mean separation distance for tracks that overlap between IBTrACS tracks and DL tracks
    perc_overlap_matrix_ibtracs_dl = np.zeros((len(ibtracs_tracks), len(dl_tracks)))
    dist_matrix_ibtracs_dl = np.zeros((len(ibtracs_tracks), len(dl_tracks)))
    dist_matrix_ibtracs_dl[:, :] = 1e6
    
    for ib_track_i, ib_track in enumerate(ibtracs_tracks):
        for dl_track_i, dl_track in enumerate(dl_tracks):
            mean_dist, perc_overlap = get_dist_between_tracks(ib_track, dl_track)
            if perc_overlap > 0:
                dist_matrix_ibtracs_dl[ib_track_i, dl_track_i] = mean_dist
                perc_overlap_matrix_ibtracs_dl[ib_track_i, dl_track_i] = perc_overlap
                
    #Calculate the percentage overlap and mean separation distance for tracks that overlap between T-/TRACK tracks and DL tracks
    perc_overlap_matrix_track_dl = np.zeros((len(track_tracks), len(dl_tracks)))
    dist_matrix_track_dl = np.zeros((len(track_tracks), len(dl_tracks)))
    dist_matrix_track_dl[:, :] = 1e6
    
    for track_track_i, track_track in enumerate(track_tracks):
        for dl_track_i, dl_track in enumerate(dl_tracks):
            mean_dist, perc_overlap = get_dist_between_tracks(track_track, dl_track)
            if perc_overlap > 0:
                dist_matrix_track_dl[track_track_i, dl_track_i] = mean_dist
                perc_overlap_matrix_track_dl[track_track_i, dl_track_i] = perc_overlap
                
    #Calculate the percentage overlap and mean separation distance for tracks that overlap between all_methods
    perc_overlap_matrix_all = np.zeros((len(ibtracs_tracks), len(track_tracks), len(dl_tracks)))
    dist_matrix_all = np.zeros((len(ibtracs_tracks), len(track_tracks), len(dl_tracks)))
    dist_matrix_all[:, :, :] = 1e6
    
    track_counter = 0
    for ib_track_i, ib_track in enumerate(ibtracs_tracks):
        for track_track_i, track_track in enumerate(track_tracks):
            for dl_track_i, dl_track in enumerate(dl_tracks):
                track_counter += 1
                #if (track_counter) % 10000 == 0:
                #    print("Processing track", track_counter, "of", len(ibtracs_tracks) * len(track_tracks) * len(dl_tracks))
                mean_dist, perc_overlap = get_dist_between_three_tracks(ib_track, track_track, dl_track)
                if perc_overlap > 0:
                    dist_matrix_all[ib_track_i, track_track_i, dl_track_i] = mean_dist
                    perc_overlap_matrix_all[ib_track_i, track_track_i, dl_track_i] = perc_overlap
    
    tr_sec = 0
    ib_sec = 0
    dl_sec = 0
    tr_ib_sec = 0
    tr_dl_sec = 0
    ib_dl_sec = 0
    all_sec = 0
    
    diff_x_ib_dl = []
    diff_y_ib_dl = []
    diff_x_ib_track = []
    diff_y_ib_track = []
    diff_x_track_dl = []
    diff_y_track_dl = []
    
    diff_x_all_ib_dl = []
    diff_y_all_ib_dl = []
    diff_x_all_ib_track = []
    diff_y_all_ib_track = []
    diff_x_all_track_dl = []
    diff_y_all_track_dl = []
    
    #Go through each combination of three for lowest mean distance diff
    while True:
        
        #Get combination of tracks that have least mean separation disstance
        ib_i, track_i, dl_i = np.unravel_index(np.argmin(dist_matrix_all), dist_matrix_all.shape)
        
        #If distance >= initialising distance, all viable combinations for all three tracks have been found
        if dist_matrix_all[ib_i, track_i, dl_i] >= 1e6:
            break
            
        #If distance is greater than 5deg (geodesic) or overlap is < 10% for base track, track is not viable, so reset its distance value and skip
        if dist_matrix_all[ib_i, track_i, dl_i] >= 5 or perc_overlap_matrix_all[ib_i, track_i, dl_i] < 10:
            dist_matrix_all[ib_i, track_i, dl_i] = 1e6
            
        #If viable
        else:
            
            #Get differences
            new_diff_x_ib_dl, new_diff_y_ib_dl, new_diff_x_ib_track, new_diff_y_ib_track, new_diff_x_track_dl, new_diff_y_track_dl, new_diff_x_all_ib_dl, new_diff_y_all_ib_dl, new_diff_x_all_ib_track, new_diff_y_all_ib_track, new_diff_x_all_track_dl, new_diff_y_all_track_dl = get_diff_for_overlapping_points_3way(ibtracs_tracks[ib_i], track_tracks[track_i], dl_tracks[dl_i], diff_x_ib_dl, diff_y_ib_dl, diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl)
            
            tr_ib_sec += len(new_diff_x_ib_track) - len(diff_x_ib_track)
            tr_dl_sec += len(new_diff_x_track_dl) - len(diff_x_track_dl)
            ib_dl_sec += len(new_diff_x_ib_dl) - len(diff_x_ib_dl)
            all_sec += len(new_diff_x_all_ib_track) - len(diff_x_all_ib_track)
            tr_sec += len(track_tracks[track_i]) - (len(new_diff_x_all_ib_track) - len(diff_x_all_ib_track)) - (len(new_diff_x_ib_track) - len(diff_x_ib_track)) - (len(new_diff_x_track_dl) - len(diff_x_track_dl))
            ib_sec += len(ibtracs_tracks[ib_i]) - (len(new_diff_x_all_ib_track) - len(diff_x_all_ib_track)) - (len(new_diff_x_ib_track) - len(diff_x_ib_track)) - (len(new_diff_x_ib_dl) - len(diff_x_ib_dl))
            dl_sec += len(dl_tracks[dl_i]) - (len(new_diff_x_all_ib_track) - len(diff_x_all_ib_track)) - (len(new_diff_x_track_dl) - len(diff_x_track_dl)) - (len(new_diff_x_ib_dl) - len(diff_x_ib_dl))
            diff_x_ib_dl, diff_y_ib_dl, diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl = new_diff_x_ib_dl, new_diff_y_ib_dl, new_diff_x_ib_track, new_diff_y_ib_track, new_diff_x_track_dl, new_diff_y_track_dl, new_diff_x_all_ib_dl, new_diff_y_all_ib_dl, new_diff_x_all_ib_track, new_diff_y_all_ib_track, new_diff_x_all_track_dl, new_diff_y_all_track_dl
            
            #Reset distance for combinations in all sources matrix so that each track can only be used once
            dist_matrix_all[ib_i, :, :] = 1e6
            dist_matrix_all[:, track_i, :] = 1e6
            dist_matrix_all[:, :, dl_i] = 1e6
            
            #Reset distance for combinations in two sources matrices so that each track can only be used once
            dist_matrix_ibtracs_track[ib_i, :] = 1e6
            dist_matrix_ibtracs_track[:, track_i] = 1e6
            
            dist_matrix_track_dl[track_i, :] = 1e6
            dist_matrix_track_dl[:, dl_i] = 1e6
            
            dist_matrix_ibtracs_dl[ib_i, :] = 1e6
            dist_matrix_ibtracs_dl[:, dl_i] = 1e6
    
    #Go through each combination of three for lowest mean distance diff
    while True:
        
        #If smallest mean separation distance is from a combination of IBTrACS and T-/TRACK tracks
        if np.argmin((np.min(dist_matrix_ibtracs_track), np.min(dist_matrix_track_dl), np.min(dist_matrix_ibtracs_dl))) == 0:
            
            #Get combination
            ib_tr_ib_i, ib_tr_tr_i = np.unravel_index(np.argmin(dist_matrix_ibtracs_track), dist_matrix_ibtracs_track.shape)
            
            #If distance >= initialising distance, all viable combinations for all combinations of tracks have been found
            if dist_matrix_ibtracs_track[ib_tr_ib_i, ib_tr_tr_i] >= 1e6:
                break
            
            #If distance is less than 5deg (geodesic) or overlap is > 10% for base track, track is viable
            if perc_overlap_matrix_ibtracs_track[ib_tr_ib_i, ib_tr_tr_i] > 10 and dist_matrix_ibtracs_track[ib_tr_ib_i, ib_tr_tr_i] < 5:

                #Get differences
                new_diff_x_ib_track, new_diff_y_ib_track = get_diff_for_overlapping_points(ibtracs_tracks[ib_tr_ib_i], track_tracks[ib_tr_tr_i], diff_x_ib_track, diff_y_ib_track)
                
                tr_ib_sec += len(new_diff_x_ib_track) - len(diff_x_ib_track)
                tr_sec += len(track_tracks[ib_tr_tr_i]) - (len(new_diff_x_ib_track) - len(diff_x_ib_track))
                ib_sec += len(ibtracs_tracks[ib_tr_ib_i]) - (len(new_diff_x_ib_track) - len(diff_x_ib_track))

                diff_x_ib_track, diff_y_ib_track = new_diff_x_ib_track, new_diff_y_ib_track
                
                #Reset distance for combinations in two sources matrices so that each track can only be used once
                dist_matrix_ibtracs_dl[ib_tr_ib_i, :] = 1e6
                dist_matrix_track_dl[ib_tr_tr_i, :] = 1e6
                dist_matrix_ibtracs_track[ib_tr_ib_i, :] = 1e6
                dist_matrix_ibtracs_track[:, ib_tr_tr_i] = 1e6

            #If not viable, reset distance
            else:
                dist_matrix_ibtracs_track[ib_tr_ib_i, ib_tr_tr_i] = 1e6

        #If smallest mean separation distance is from a combination of T-/TRACK and DL tracks
        elif np.argmin((np.min(dist_matrix_ibtracs_track), np.min(dist_matrix_track_dl), np.min(dist_matrix_ibtracs_dl))) == 1:
            
            #Get combination
            tr_dl_tr_i, tr_dl_dl_i = np.unravel_index(np.argmin(dist_matrix_track_dl), dist_matrix_track_dl.shape)
            
            #If distance >= initialising distance, all viable combinations for all combinations of tracks have been found
            if dist_matrix_track_dl[tr_dl_tr_i, tr_dl_dl_i] >= 1e6:
                break
            
            #If distance is less than 5deg (geodesic) or overlap is > 10% for base track, track is viable
            if perc_overlap_matrix_track_dl[tr_dl_tr_i, tr_dl_dl_i] > 10 and dist_matrix_track_dl[tr_dl_tr_i, tr_dl_dl_i] < 5:
                
                #Get differences
                new_diff_x_track_dl, new_diff_y_track_dl = get_diff_for_overlapping_points(track_tracks[tr_dl_tr_i], dl_tracks[tr_dl_dl_i], diff_x_track_dl, diff_y_track_dl)
                
                tr_dl_sec += len(new_diff_x_track_dl) - len(diff_x_track_dl)
                tr_sec += len(track_tracks[tr_dl_tr_i]) - (len(new_diff_x_track_dl) - len(diff_x_track_dl))
                dl_sec += len(dl_tracks[tr_dl_dl_i]) - (len(new_diff_x_track_dl) - len(diff_x_track_dl))
                
                diff_x_track_dl, diff_y_track_dl = new_diff_x_track_dl, new_diff_y_track_dl
                
                #Reset distance for combinations in two sources matrices so that each track can only be used once
                dist_matrix_ibtracs_track[:, tr_dl_tr_i] = 1e6
                dist_matrix_ibtracs_dl[:, tr_dl_dl_i] = 1e6
                dist_matrix_track_dl[tr_dl_tr_i, :] = 1e6
                dist_matrix_track_dl[:, tr_dl_dl_i] = 1e6
                
            #If not viable, reset distance
            else:
                dist_matrix_track_dl[tr_dl_tr_i, tr_dl_dl_i] = 1e6
                
        #If smallest mean separation distance is from a combination of IBTrACS and DL tracks
        else:
            
            #Get combination
            ib_dl_ib_i, ib_dl_dl_i = np.unravel_index(np.argmin(dist_matrix_ibtracs_dl), dist_matrix_ibtracs_dl.shape)
            
            #If distance >= initialising distance, all viable combinations for all combinations of tracks have been found
            if dist_matrix_ibtracs_dl[ib_dl_ib_i, ib_dl_dl_i] >= 1e6:
                break
            
            #If distance is less than 5deg (geodesic) or overlap is > 10% for base track, track is viable
            if perc_overlap_matrix_ibtracs_dl[ib_dl_ib_i, ib_dl_dl_i] > 10 and dist_matrix_ibtracs_dl[ib_dl_ib_i, ib_dl_dl_i] < 5:
                
                #Get differences
                new_diff_x_ib_dl, new_diff_y_ib_dl = get_diff_for_overlapping_points(ibtracs_tracks[ib_dl_ib_i], dl_tracks[ib_dl_dl_i], diff_x_ib_dl, diff_y_ib_dl)
                
                ib_dl_sec += len(new_diff_x_ib_dl) - len(diff_x_ib_dl)
                ib_sec += len(ibtracs_tracks[ib_dl_ib_i]) - (len(new_diff_x_ib_dl) - len(diff_x_ib_dl))
                dl_sec += len(dl_tracks[ib_dl_dl_i]) - (len(new_diff_x_ib_dl) - len(diff_x_ib_dl))
                
                diff_x_ib_dl, diff_y_ib_dl = new_diff_x_ib_dl, new_diff_y_ib_dl
                
                #Reset distance for combinations in two sources matrices so that each track can only be used once
                dist_matrix_ibtracs_track[ib_dl_ib_i, :] = 1e6
                dist_matrix_track_dl[:, ib_dl_dl_i] = 1e6
                dist_matrix_ibtracs_dl[ib_dl_ib_i, :] = 1e6
                dist_matrix_ibtracs_dl[:, ib_dl_dl_i] = 1e6
            
            #If not viable, reset distance
            else:
                dist_matrix_ibtracs_dl[ib_dl_ib_i, ib_dl_dl_i] = 1e6
           
    venn_sections = [tr_sec, ib_sec, dl_sec, tr_ib_sec, tr_dl_sec, ib_dl_sec, all_sec]
    
    return venn_sections, diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_ib_dl, diff_y_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl

def get_dist_between_tracks(base_track: list[list[dt, float, float]], track: list[list[dt, float, float]]) -> tuple(int, float):

    '''
    Function to calculate the mean separation distance for two tracks and the % of overlap for the base track
    args:
        base_track: list of TC centres [date / lat / lon] making up a track; will be the track to compare the other to
        track: list of TC centres [date / lat / lon] making up a track; will be compared to base_track
    returns:
        mean_dist: mean separation distance in degrees (geodesic)
        perc_overlap: percentage for overlap from base_track
    '''
    
    # Initialise counters
    dist = 0
    overlap_pts = 0
    
    # Go through points in base_track
    for base_pt in base_track:
        
        # Get date, lat, lon from base track 
        if len(base_pt) == 6:
            base_date, _, base_lat, base_lon, _, _ = base_pt
        else:
            base_date, base_lat, base_lon = base_pt
        
        if len(track) > 0:
            # Find point from track that has the same date/time of point from base_track
            for track_pt in track:

                if len(track_pt) == 6:
                    track_date, _, track_lat, track_lon, _, _ = track_pt
                else:
                    track_date, track_lat, track_lon = track_pt

                if base_date == track_date:
                    break

            # If point from base_track and track have the same date/time, update separation distance and overlap variables
            if base_date == track_date:
                dist += geodesic_distance(base_lon, base_lat, track_lon, track_lat)
                overlap_pts += 1

    # If any overlap between tracks, return mean separation distance and % of overlap calculated on base_track
    if overlap_pts > 0:
        mean_dist = dist / overlap_pts
        perc_overlap = overlap_pts / len(base_track) * 100
        return mean_dist, perc_overlap
    
    # Else return 1e6 as mean separation distance and 0% overlap
    else:
        return 1e6, 0
    
def get_dist_between_three_tracks(base_track: list[list[dt, float, float]], track1: list[list[dt, float, float]], track2: list[list[dt, float, float]]) -> tuple(float, float):
    
    '''
    Function to calculate the mean separation distance for the overlapping parts of three tracks and the % of overlap for the base track
    args:
        base_track: list of TC centres [date / lat / lon] making up a track; will be the track to compare the other to
        track1: list of TC centres [date / lat / lon] making up a track; will be compared to base_track and track2
        track2: list of TC centres [date / lat / lon] making up a track; will be compared to base_track and track1
    returns:
        mean_dist: mean separation distance in degrees (geodesic)
        perc_overlap: percentage for overlap from base_track
    '''
    
    # Initialise counters
    dist = 0
    overlap_pts = 0
    
    # Go through points in base_track
    for base_pt in base_track:
        
        # Get date, lat, lon from base track 
        if len(base_pt) == 6:
            base_date, _, base_lat, base_lon, _, _ = base_pt
        else:
            base_date, base_lat, base_lon = base_pt
            
        # Find point from track1 that has the same date/time of point from base_track
        for track1_pt in track1:
            
            if len(track1_pt) == 6:
                track1_date, _, track1_lat, track1_lon, _, _ = track1_pt
            else:
                track1_date, track1_lat, track1_lon = track1_pt
            
            # If point from base_track and track1 have the same date/time, update separation distance and overlap variables
            if base_date == track1_date:
                break
        
        # Find point from track2 that has the same date/time of point from base_track
        for track2_pt in track2:
            
            if len(track2_pt) == 6:
                track2_date, _, track2_lat, track2_lon, _, _ = track2_pt
            else:
                track2_date, track2_lat, track2_lon = track2_pt
            
            # If point from base_track and track2 have the same date/time, update separation distance and overlap variables
            if base_date == track2_date:
                break
                
        # If same date, increment distance by max distance between points
        if base_date == track1_date and base_date == track2_date:
            dist += max(geodesic_distance(base_lon, base_lat, track1_lon, track1_lat), geodesic_distance(base_lon, base_lat, track2_lon, track2_lat), geodesic_distance(track1_lon, track1_lat, track2_lon, track2_lat))
            overlap_pts += 1
            
    # If any overlap between tracks, return mean separation distance and % of overlap calculated on base_track
    if overlap_pts > 0:
        mean_dist = dist / overlap_pts
        perc_overlap = overlap_pts / len(base_track) * 100
        return mean_dist, perc_overlap
    
    # Else return 1e6 as mean separation distance and 0% overlap
    else:
        return 1e6, 0

def get_diff_for_overlapping_points_3way(ibtracs_track: dict, track_track: dict, dl_track: dict, diff_x_ib_dl: list[float], diff_y_ib_dl: list[float], diff_x_ib_track: list[float], diff_y_ib_track: list[float], diff_x_track_dl: list[float], diff_y_track_dl: list[float], diff_x_all_ib_dl: list[float], diff_y_all_ib_dl: list[float], diff_x_all_ib_track: list[float], diff_y_all_ib_track: list[float], diff_x_all_track_dl: list[float], diff_y_all_track_dl: list[float]) -> tuple(list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]):

    '''
    Function to calculate the difference between lats and lons for overlapping points of three tracks, split up by Venn diagram section. If only two tracks overlap, the difference is stored in the relevant Venn diagram section
    args:
        ibtracs_dict: dict of TC centres as cenerated by IBTrACS with keys being datetime objects
        track_dict: dict of TC centres as cenerated by T-/TRACK with keys being datetime objects
        dl_dict: dict of TC centres as cenerated by the Deep Learning model with keys being datetime objects
        diff_x_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_y_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_x_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_ib_track: list of differences in latgitude, i.e. lat_ib - lat_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_track_dl: list of differences in longitude, i.e. lon_track - lon_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        diff_y_track_dl: list of differences in latitude, i.e. lat_track - lat_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        diff_x_all_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_y_all_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_x_all_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_all_ib_track: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_all_track_dl: list of differences in longitude, i.e. lon_tr - lon_dl, for already processed points at which all sources overlap
        diff_y_all_track_dl: list of differences in latitude, i.e. lat_tr - lat_dl, for already processed points at which all sources overlap
    returns:
        new_diff_x_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        new_diff_y_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        new_diff_x_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        new_diff_y_ib_track: list of differences in latgitude, i.e. lat_ib - lat_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        new_diff_x_track_dl: list of differences in longitude, i.e. lon_track - lon_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        new_diff_y_track_dl: list of differences in latitude, i.e. lat_track - lat_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        new_diff_x_all_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        new_diff_y_all_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        new_diff_x_all_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        new_diff_y_all_ib_track: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        new_diff_x_all_track_dl: list of differences in longitude, i.e. lon_tr - lon_dl, for already processed points at which all sources overlap
        new_diff_y_all_track_dl: list of differences in latitude, i.e. lat_tr - lat_dl, for already processed points at which all sources overlap
    '''
    
    new_diff_x_ib_dl, new_diff_y_ib_dl, new_diff_x_ib_track, new_diff_y_ib_track, new_diff_x_track_dl, new_diff_y_track_dl, new_diff_x_all_ib_dl, new_diff_y_all_ib_dl, new_diff_x_all_ib_track, new_diff_y_all_ib_track, new_diff_x_all_track_dl, new_diff_y_all_track_dl = copy.deepcopy([diff_x_ib_dl, diff_y_ib_dl, diff_x_ib_track, diff_y_ib_track, diff_x_track_dl, diff_y_track_dl, diff_x_all_ib_dl, diff_y_all_ib_dl, diff_x_all_ib_track, diff_y_all_ib_track, diff_x_all_track_dl, diff_y_all_track_dl])
    
    # Initialise list to hold all dates present in tracks
    dates_in_tracks = []
    
    # Get all dates present in IBTrACS track
    for ib_pt in ibtracs_track:
        ib_date, lat, lon = ib_pt
        dates_in_tracks.append(ib_date)
        
    # Get all dates present in T-/TRACK track
    for tr_pt in track_track:
        tr_date, tr_lat, tr_lon = tr_pt
        if tr_date not in dates_in_tracks:
            dates_in_tracks.append(tr_date)
                
    # Get all dates present in DL track
    for dl_pt in dl_track:
        dl_date, dl_lat, dl_lon = dl_pt
        if dl_date not in dates_in_tracks:
            dates_in_tracks.append(dl_date)
            
    # Go through each date
    for date in dates_in_tracks:
        
        # Get point at date from IBTrACS track
        for ib_pt in ibtracs_track:
            ib_date, ib_lat, ib_lon = ib_pt
            if ib_date == date:
                break
        
        # Get point at date from T-/TRACK track
        for tr_pt in track_track:
            tr_date, tr_lat, tr_lon = tr_pt
            if tr_date == date:
                break
        
        # Get point at date from DL track
        for dl_pt in dl_track:
            dl_date, dl_lat, dl_lon = dl_pt
            if dl_date == date:
                break
                
        # If all tracks have point at date
        if ib_date == date and dl_date == date and tr_date == date:
            
            # Calculate and store difference between IBTrACS and DL points
            new_diff_x_all_ib_dl.append(ib_lon - dl_lon)
            new_diff_y_all_ib_dl.append(ib_lat - dl_lat)
            
            # Calculate and store difference between IBTrACS and T-/TRACK points
            new_diff_x_all_ib_track.append(ib_lon - tr_lon)
            new_diff_y_all_ib_track.append(ib_lat - tr_lat)
            
            # Calculate and store difference between T-/TRACK and DL points
            new_diff_x_all_track_dl.append(tr_lon - dl_lon)
            new_diff_y_all_track_dl.append(tr_lat - dl_lat)
        
        # If only IBTrACS and T-/TRACK tracks have a point at date, calculate and store difference
        elif ib_date == date and tr_date == date and dl_date != date:
            new_diff_x_ib_track.append(ib_lon - tr_lon)
            new_diff_y_ib_track.append(ib_lat - tr_lat)
            
        # If only IBTrACS and DL tracks have a point at date, calculate and store difference
        elif ib_date == date and tr_date != date and dl_date == date:
            new_diff_x_ib_dl.append(ib_lon - dl_lon)
            new_diff_y_ib_dl.append(ib_lat - dl_lat)
            
        # If only T-/TRACK and DL tracks have a point at date, calculate and store difference
        elif ib_date != date and tr_date == date and dl_date == date:
            new_diff_x_track_dl.append(tr_lon - dl_lon)
            new_diff_y_track_dl.append(tr_lat - dl_lat)
        
    return new_diff_x_ib_dl, new_diff_y_ib_dl, new_diff_x_ib_track, new_diff_y_ib_track, new_diff_x_track_dl, new_diff_y_track_dl, new_diff_x_all_ib_dl, new_diff_y_all_ib_dl, new_diff_x_all_ib_track, new_diff_y_all_ib_track, new_diff_x_all_track_dl, new_diff_y_all_track_dl

def get_diff_for_overlapping_points(base_track: list, track: list, diff_x: list[float], diff_y: list[float]) -> tuple(list[float], list[float]):

    '''
    Function to calculate the difference between lats and lons for overlapping points of two tracks
    args:
        base_track: track to compare other to
        track: track to be compared to base_track
        diff_x: difference in longitude between overlapping points of the tracks
        diff_y: difference in latitude between overlapping points of the track
    returns:
        new_diff_x: difference in longitude between overlapping points of the tracks
        new_diff_y: difference in latitude between overlapping points of the track
    '''
    
    new_diff_x = copy.deepcopy(diff_x)
    new_diff_y = copy.deepcopy(diff_y)
    
    # Initialise list to hold all dates present in tracks
    dates_in_tracks = []
    
    # Get all dates present in base_track
    for pt in base_track:
        if len(pt) == 6:
            date, _, lat, lon, _, _ = pt
        else:
            date, lat, lon = pt
        if date not in dates_in_tracks:
            dates_in_tracks.append(date)
     
    # Get all dates present in track
    for pt in track:
        if len(pt) == 6:
            date, _, lat, lon, _, _ = pt
        else:
            date, lat, lon = pt
        if date not in dates_in_tracks:
            dates_in_tracks.append(date)
    
    # For each date
    for date in dates_in_tracks:
        
        # Get point with same date/time from base_track
        for base_pt in base_track:
            if len(base_pt) == 6:
                base_date, _, base_lat, base_lon, _, _ = base_pt
            else:
                base_date, base_lat, base_lon = base_pt
            if base_date == date:
                break
          
        # Get point with same date/time from track
        for track_pt in track:
            if len(track_pt) == 6:
                track_date, _, track_lat, track_lon, _, _ = track_pt
            else:
                track_date, track_lat, track_lon = track_pt
            if track_date == date:
                break
                
        # If date/time in both tracks, calculatr and store differences
        if base_date == date and track_date == date:
            new_diff_x.append(base_lon - track_lon)
            new_diff_y.append(base_lat - track_lat)
            
    return new_diff_x, new_diff_y

def plot_correlations(diff_x_ib_track: list[float], diff_y_ib_track: list[float], diff_x_track_dl: list[float], diff_y_track_dl: list[float], diff_x_ib_dl: list[float], diff_y_ib_dl: list[float], diff_x_all_ib_track: list[float], diff_y_all_ib_track: list[float], diff_x_all_track_dl: list[float], diff_y_all_track_dl: list[float], diff_x_all_ib_dl: list[float], diff_y_all_ib_dl: list[float], max_diff: float, track_name: str, save_path: str) -> None:   

    '''
    Function to plot correlations between tracks of different sources.
    args:
        diff_x_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_y_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_x_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_ib_track: list of differences in latgitude, i.e. lat_ib - lat_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_track_dl: list of differences in longitude, i.e. lon_track - lon_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        diff_y_track_dl: list of differences in latitude, i.e. lat_track - lat_dl, for already processed points at which only T-/TRACK and DL tracks overlap
        diff_x_all_ib_dl: list of differences in longitude, i.e. lon_ib - lon_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_y_all_ib_dl: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and DL tracks overlap
        diff_x_all_ib_track: list of differences in longitude, i.e. lon_ib - lon_track, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_y_all_ib_track: list of differences in latitude, i.e. lat_ib - lat_dl, for already processed points at which only IBTrACS and T-/TRACK tracks overlap
        diff_x_all_track_dl: list of differences in longitude, i.e. lon_tr - lon_dl, for already processed points at which all sources overlap
        diff_y_all_track_dl: list of differences in latitude, i.e. lat_tr - lat_dl, for already processed points at which all sources overlap
        max_diff: extent to whihc to clip axes of plots
        track_name: name of TRACK technique used: TRACK or T-/TRACK
        save_path: name to use when saving figure
    returns:
        None
    '''
    
    # Set up subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(3*5.5, 2*5))
    
    axbig = axs[0][0]
    
    # Plot differences for points in IBTrACS and T-/TRACK
    axbig.plot(-np.array(diff_x_ib_track), -np.array(diff_y_ib_track), "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title(track_name + " vs IBTrACS")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Subplot regridding
    axbig = axs[0][1]
    
    # Plot differences for points in T-/TRACK and DL
    axbig.plot(diff_x_track_dl, diff_y_track_dl, "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title(track_name + " vs DL Model")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Subplot regridding
    axbig = axs[0][2]
    
    # Plot differences for points in IBTrACS and DL
    axbig.plot(diff_x_ib_dl, diff_y_ib_dl, "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title("IBTrACS vs DL Model")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Plot differences for points in IBTrACS and T-/TRACK but were found in all sources
    axbig = axs[1][0]
    axbig.plot(-np.array(diff_x_all_ib_track), -np.array(diff_y_all_ib_track), "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title("All Methods - " + track_name + " vs IBTrACS")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Plot differences for points in T-/TRACK and DL but were found in all sources
    axbig = axs[1][1]
    axbig.plot(diff_x_all_track_dl, diff_y_all_track_dl, "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title("All Methods - " + track_name + " vs DL Model")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Plot differences for points in IBTrACS and DL but were found in all sources
    axbig = axs[1][2]
    axbig.plot(diff_x_all_ib_dl, diff_y_all_ib_dl, "o")
    axbig.set_xlim([-max_diff, max_diff])
    axbig.set_ylim([-max_diff, max_diff])
    axbig.axes.set_aspect('equal')
    axbig.set_title("All Methods - IBTrACS vs DL Model")
    axbig.axvline(x=0, color="k", alpha=0.3, ls="--")
    axbig.axhline(y=0, color="k", alpha=0.3, ls="--")
    axbig.set_xlabel("Diff in Longitude")
    axbig.set_ylabel("Diff in Latitude")
    
    # Save figure
    plt.savefig(save_path)

def track_in_month(track: list[list[dt, float, float]], year: int, month: int, region: int) -> bool:

    '''
    Check if the given track is present in a given month for a specific region
    args:
        track: list of TC points making up a track
        year: year to search in
        month: month to search in
        region: region id to search in
    '''
    
    # get lat/lon bounds
    if region == 0:
        min_lat = -90
        max_lat = 90
        min_lon = -180
        max_lon = 180
    elif 1 <= region < 5:
        min_lat = -60
        max_lat = 0
        if region == 1:
            min_lon = 20
            max_lon = 100
        elif region == 2:
            min_lon = 100
            max_lon = 180
        elif region == 3:
            min_lon = -180
            max_lon = -100
        elif region == 4:
            min_lon = -100
            max_lon = -20
    elif 5 <= region < 9:
        min_lat = 0
        max_lat = 60
        if region == 5:
            min_lon = 20
            max_lon = 100
        elif region == 6:
            min_lon = 100
            max_lon = 180
        elif region == 7:
            min_lon = -180
            max_lon = -100
        elif region == 8:
            min_lon = -100
            max_lon = -20
         
    # for each point in the track, check if point is in month and return True if so
    for pt in track:
        date, lat, lon = pt
        
        if date.year == year and date.month == month and min_lat < lat < max_lat and min_lon < lon < max_lon:
            return True
        
    # return false if not
    return False

def get_frequencies(ibtracs_tracks: list[list[dt, float, float]], t_track_tracks: list[list[dt, float, float]], track_tracks: list[list[dt, float, float]], dl_tracks: list[list[dt, float, float]], region: int) -> tuple(list[int], list[int], list[int], list[int]):
    
    '''
    Get monthly TC frequencies
    args:
        ibtracs_tracks: list of IBTrACS tracks
        t_track_tracks: list of T-TRACK tracks
        track_tracks: list of TRACK tracks
        dl_tracks: list of DL tracks
        region: region to get frequencies for
    returns:
        ib_freq: number of TCs in IBTrACS
        t_track_freq: number of TCs in T-TRACK
        track_freq: number of TCs in TRACK
        dl_freq: number of TCs in DL
    '''
    
    # set up lists
    ib_freq = []
    t_track_freq = []
    track_freq = []
    dl_freq = []
    
    # for each year in test dataset
    for year in [2017, 2018, 2019]:
        
        # get time bounds
        if year == 2017:
            month_start = 8
            month_end = 12
        elif year == 2018:
            month_start = 1
            month_end = 12
        elif year == 2019:
            month_start = 1
            month_end = 8
            
        # for each month
        for month in range(month_start, month_end+1):
            
            # get IBTrACS frequencies
            ib_tracks_month = []
            for track in ibtracs_tracks:
                if track_in_month(track, year, month, region) == True:
                    ib_tracks_month.append(track)
            ib_freq.append(len(ib_tracks_month))
            
            # get T-TRACK frequencies
            t_track_tracks_month = []
            for track in t_track_tracks:
                if track_in_month(track, year, month, region) == True:
                    t_track_tracks_month.append(track)
            t_track_freq.append(len(t_track_tracks_month))
            
            # get TRACK frequencies
            track_tracks_month = []
            for track in track_tracks:
                if track_in_month(track, year, month, region) == True:
                    track_tracks_month.append(track)
            track_freq.append(len(track_tracks_month))
            
            # get TCDetect frequencies
            dl_tracks_month = []
            for track in dl_tracks:
                if track_in_month(track, year, month, region) == True:
                    dl_tracks_month.append(track)
            dl_freq.append(len(dl_tracks_month))
            
    return ib_freq, t_track_freq, track_freq, dl_freq
            
def plot_frequencies(ib_freq: list[float], t_track_freq: list[float], dl_freq: list[float], save_path: str = "freq.pdf") -> None:

    '''
    Plot TC monthly frequencies
    args:
        ib_freq: IBTrACS monthly frequencies
        t_track_freq: T-TRACK monthly frequencies
        dl_freq: TCDetect monthly frequencies
        save_path: path to save figure at
    '''
    
    fig, ax = plt.subplots(9, 1, figsize=(10, 10))
    
    for i in range(9):
        
        ax[i].plot(ib_freq[i], color="r", label="IBTrACS")
        ax[i].plot(t_track_freq[i], color="b", label="T-TRACK")
        ax[i].plot(dl_freq[i], color="g", label="TCDetect")
        ax[i].set_yticks([])
        
        if i != 8:
            ax[i].set_xticks([])
            
        if i == 0:
            ax[i].set_ylabel("Whole World", rotation=0, labelpad=60)
        elif i == 1:
            ax[i].set_ylabel("South Indian Ocean", rotation=0, labelpad=60)
        elif i == 2:
            ax[i].set_ylabel("SW Pacific", rotation=0, labelpad=60)
        elif i == 3:
            ax[i].set_ylabel("SE Pacific", rotation=0, labelpad=60)
        elif i == 4:
            ax[i].set_ylabel("Southern Atlantic", rotation=0, labelpad=60)
        elif i == 5:
            ax[i].set_ylabel("North Indian Ocean", rotation=0, labelpad=60)
        elif i == 6:
            ax[i].set_ylabel("NW Pacific", rotation=0, labelpad=60)
        elif i == 7:
            ax[i].set_ylabel("NE Pacific", rotation=0, labelpad=60)
        elif i == 8:
            ax[i].set_ylabel("North Atlantic", rotation=0, labelpad=60)
        
    ax[-1].set_xlabel("Months since 01-08-2017")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[4].legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0))
    plt.tight_layout()
    plt.savefig(save_path)
    
def plot_all_points(ib: dict, tr: dict, dl: dict, name: str) -> None:

    '''
    Function to plot all TC centres given by all three methods.
    args:
        ib: dict of TC centres as cenerated by IBTrACS with keys being datetime objects
        tr: dict of TC centres as cenerated by T-/TRACK with keys being datetime objects
        dl: dict of TC centres as cenerated by the Deep Learning model with keys being datetime objects
        name: Name to be used when saving figure
    returns:
        None
    '''
    
    # Set up region box for Basemap
    lonmin = -180
    lonmax = 180
    latmin = -90
    latmax = 90
    
    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2*10, 2*5))
    
    # Get first subplot axes
    axbig = axes[0, 0]
    
    # Create lists of IBTrACS TC centres' lat/lon combination
    lats = []
    lons = []
    for key in ib.keys():
        for storm in ib[key]:
            lat, lon = storm
            lats.append(lat)
            lons.append(lon)
            
    # Plot points
    m = Basemap(llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, ax=axbig)
    m.drawcoastlines(color='k',linewidth=0.5,zorder=3)
    m.drawcountries(color='k',linewidth=0.1,zorder=3)
    x, y = m(lons, lats)
    m.plot(x, y,'o',ms=2, zorder=4)
    
    # Get second subplot axes
    axbig = axes[0, 1]
    
    # Create lists of T-/TRACK TC centres' lat/lon combination
    lats = []
    lons = []
    for key in tr.keys():
        for storm in tr[key]:
            lat, lon = storm
            lats.append(lat)
            lons.append(lon)
            
    # Plot points
    m = Basemap(llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, ax=axbig)
    m.drawcoastlines(color='k',linewidth=0.5,zorder=3)
    m.drawcountries(color='k',linewidth=0.1,zorder=3)
    x, y = m(lons, lats)
    m.plot(x, y,'o',ms=2, zorder=4)
    
    # Combine subplot axes in second row to make one centered one
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :])
    
    # Create lists of DL model's TC centres' lat/lon combination
    lats = []
    lons = []
    for key in dl.keys():
        for storm in dl[key]:
            lat, lon = storm
            lats.append(lat)
            lons.append(lon)
            
    # Plot points
    m = Basemap(llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, ax=axbig)
    m.drawcoastlines(color='k',linewidth=0.5,zorder=3)
    m.drawcountries(color='k',linewidth=0.1,zorder=3)
    x, y = m(lons, lats)
    m.plot(x, y,'o',ms=2, zorder=4)
    
    # Save figure
    plt.savefig(name)

def plot_distributions(ibtracs_dict: dict, track_dict: dict, dl_dict: dict, track_name: str, name: str) -> None:

    '''
    Function to plot latitude histogram and KDE for points from all three sources
    args:
        ibtracs_dict: dict of TC centres as cenerated by IBTrACS with keys being datetime objects
        track_dict: dict of TC centres as cenerated by T-/TRACK with keys being datetime objects
        dl_dict: dict of TC centres as cenerated by the Deep Learning model with keys being datetime objects
        track_name: name of TRACK method: TRACK or T-TRACK
        name: Name to be used when saving figure
    returns:
        None
    '''
    
    # Compile lats for TCs from IBTrACS
    ib_lats = []
    for key in ibtracs_dict.keys():
        for storm in ibtracs_dict[key]:
            lat, lon = storm
            ib_lats.append(lat)
    
    # Compile lats for TCs from T-/TRACK
    track_lats = []
    for key in track_dict.keys():
        for storm in track_dict[key]:
            lat, lon = storm
            track_lats.append(lat)
    
    # Compile lats for TCs from DL
    dl_lats = []
    for key in dl_dict.keys():
        for storm in dl_dict[key]:
            lat, lon = storm
            dl_lats.append(lat)
     
    # Plotting args
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    
    # Plot
    plt.figure()
    sns.distplot(ib_lats, bins=180, label="IBTrACS", kde=True, color="b")
    sns.distplot(track_lats, bins=180, label=track_name, kde=True, color="k")
    sns.distplot(dl_lats, bins=180, label="TCDetect", kde=True, color="r")
    plt.legend()
    plt.xlabel("Latitude (deg)")
    
    # Save figure
    plt.savefig(name)

def generate_means(points: list[list[dt, float, float]], extent: int, pool: mp.Pool) -> np.array:

    '''
    Function to generate composite cases.
    args:
        points: list of TC centres, including their date/time
        extent: length of bounding box to be centred at TC centre for which to get data
        pool: multiprocessing pool
    returns:
        mean: numpy array of shape (43, 43, 5) containing composite case
    '''
    
    # Setup inputs for loading cases using multiprocessing pool
    inputs = []
    for pt in points:
        date, lat, lon = pt
        inputs.append([date, lat, lon, extent])
        
    # Load data
    results = list(pool.map(load_nc_file, inputs))
    
    # Go through all cases and sotre in one array of same size
    data = np.zeros((len(results), 43, 43, 5))
    for res_i, res in enumerate(results):
        res_data, _, _ = res
        for channel in range(5):
            data[res_i, :, :, channel] = cv2.resize(res_data[:, :, channel], (43, 43))
          
    # Take mean
    mean = np.zeros(data.shape[1:])
    mean[:, :, 0] = np.mean(data[:, :, :, 0], axis=0)
    mean[:, :, 1] = np.mean(data[:, :, :, 1], axis=0)
    mean[:, :, 2] = np.mean(data[:, :, :, 2], axis=0)
    mean[:, :, 3] = np.mean(data[:, :, :, 3], axis=0)
    mean[:, :, 4] = np.mean(data[:, :, :, 4], axis=0)
    mean = np.flipud(mean)
    
    return mean  

def plot_means(ibtracs_dict: dict, track_dict: dict, dl_dict: dict, track_name: str, hemi: int, save_name: str) -> tuple(float, float, float, float, float, float, float, float, float, float, list[np.array]):
    
    '''
    Function to plot TC composite cases according to Venn diagram section.
    args:
        ibtracs_dict: dict of TC centres as cenerated by IBTrACS with keys being datetime objects
        track_dict: dict of TC centres as cenerated by T-/TRACK with keys being datetime objects
        dl_dict: dict of TC centres as cenerated by the Deep Learning model with keys being datetime objects
        track_name: Name of TRACK method used: TRACK or T-TRACK
        hemi: hemisphere for which to process TCs; 0 = all, 1 = NH, 2 = SH
        save_name: Name to use when saving figure
    returns:
        mslp_min: minimum value for MSLP colorbar
        mslp_max: maximum value for MSLP colorbar
        wind_min: minimum value for wind colorbar
        wind_max: maximum value for wind colorbar
        vort850_min: minimum value for vort850 colorbar
        vort850_max: maximum value for vort850 colorbar
        vort700_min: minimum value for vort700 colorbar
        vort700_max: maximum value for vort700 colorbar
        vort600_min: minimum value for vort600 colorbar
        vort600_max: maximum value for vort600 colorbar
        means: list of means as in plot
    '''
    
    # Get lists of TC centres for each Venn diagram section
    _, _, _, _, _, _, _, tr_points, ib_points, dl_points, tr_ib_points, tr_dl_points, ib_dl_points, all_methods_points = generate_venn_numbers(ibtracs_dict, track_dict, dl_dict, hemi)
    
    # Open pool of workers
    pool = mp.Pool(int(0.5*mp.cpu_count()))
    
    # Generate means
    print("Generating means for TRACK cases...")
    tr_mean = generate_means(tr_points, 30, pool)
    
    print("Generating means for IBTrACS cases...")
    ib_mean = generate_means(ib_points, 30, pool)
    
    print("Generating means for Deep Learning cases...")
    dl_mean = generate_means(dl_points, 30, pool)
    
    print("Generating means for TRACK and IBTrACS cases...")
    tr_ib_mean = generate_means(tr_ib_points, 30, pool)
    
    print("Generating means for TRACK and DL Model cases...")
    tr_dl_mean = generate_means(tr_dl_points, 30, pool)
    
    print("Generating means for IBTrACS and DL Model cases...")
    ib_dl_mean = generate_means(ib_dl_points, 30, pool)
    
    print("Generating means for All Methods cases...")
    all_methods_mean = generate_means(all_methods_points, 30, pool)
    
    means = [tr_mean, ib_mean, dl_mean, tr_ib_mean, tr_dl_mean, ib_dl_mean, all_methods_mean]
    
    # Close multiprocessing pool
    pool.close()
    
    # Set up subplots
    fig, axs = plt.subplots(7, 5, figsize=(5*5, 7*5))  
    
    mslp_max = np.max([tr_mean[:, :, 0].flatten(), ib_mean[:, :, 0].flatten(), dl_mean[:, :, 0].flatten(), tr_ib_mean[:, :, 0].flatten(), tr_dl_mean[:, :, 0].flatten(), ib_dl_mean[:, :, 0].flatten(), all_methods_mean[:, :, 0].flatten()])
    mslp_min = np.min([tr_mean[:, :, 0].flatten(), ib_mean[:, :, 0].flatten(), dl_mean[:, :, 0].flatten(), tr_ib_mean[:, :, 0].flatten(), tr_dl_mean[:, :, 0].flatten(), ib_dl_mean[:, :, 0].flatten(), all_methods_mean[:, :, 0].flatten()])
    wind_max = np.max([tr_mean[:, :, 1].flatten(), ib_mean[:, :, 1].flatten(), dl_mean[:, :, 1].flatten(), tr_ib_mean[:, :, 1].flatten(), tr_dl_mean[:, :, 1].flatten(), ib_dl_mean[:, :, 1].flatten(), all_methods_mean[:, :, 1].flatten()])
    wind_min = np.min([tr_mean[:, :, 1].flatten(), ib_mean[:, :, 1].flatten(), dl_mean[:, :, 1].flatten(), tr_ib_mean[:, :, 1].flatten(), tr_dl_mean[:, :, 1].flatten(), ib_dl_mean[:, :, 1].flatten(), all_methods_mean[:, :, 1].flatten()])
    vort850_max = np.max([tr_mean[:, :, 2].flatten(), ib_mean[:, :, 2].flatten(), dl_mean[:, :, 2].flatten(), tr_ib_mean[:, :, 2].flatten(), tr_dl_mean[:, :, 2].flatten(), ib_dl_mean[:, :, 2].flatten(), all_methods_mean[:, :, 2].flatten()])
    vort850_min = np.min([tr_mean[:, :, 2].flatten(), ib_mean[:, :, 2].flatten(), dl_mean[:, :, 2].flatten(), tr_ib_mean[:, :, 2].flatten(), tr_dl_mean[:, :, 2].flatten(), ib_dl_mean[:, :, 2].flatten(), all_methods_mean[:, :, 2].flatten()])
    vort700_max = np.max([tr_mean[:, :, 3].flatten(), ib_mean[:, :, 3].flatten(), dl_mean[:, :, 3].flatten(), tr_ib_mean[:, :, 3].flatten(), tr_dl_mean[:, :, 3].flatten(), ib_dl_mean[:, :, 3].flatten(), all_methods_mean[:, :, 3].flatten()])
    vort700_min = np.min([tr_mean[:, :, 3].flatten(), ib_mean[:, :, 3].flatten(), dl_mean[:, :, 3].flatten(), tr_ib_mean[:, :, 3].flatten(), tr_dl_mean[:, :, 3].flatten(), ib_dl_mean[:, :, 3].flatten(), all_methods_mean[:, :, 3].flatten()])
    vort600_max = np.max([tr_mean[:, :, 4].flatten(), ib_mean[:, :, 4].flatten(), dl_mean[:, :, 4].flatten(), tr_ib_mean[:, :, 4].flatten(), tr_dl_mean[:, :, 4].flatten(), ib_dl_mean[:, :, 4].flatten(), all_methods_mean[:, :, 4].flatten()])
    vort600_min = np.min([tr_mean[:, :, 4].flatten(), ib_mean[:, :, 4].flatten(), dl_mean[:, :, 4].flatten(), tr_ib_mean[:, :, 4].flatten(), tr_dl_mean[:, :, 4].flatten(), ib_dl_mean[:, :, 4].flatten(), all_methods_mean[:, :, 4].flatten()])
    
    # Plot composite case for TCs in TRACK only Venn section 
    style_subplots(axs[0][0], tr_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[0][1], tr_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[0][2], tr_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[0][3], tr_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[0][4], tr_mean[:, :, 4], vort600_min, vort600_max)
    axs[0][0].set_ylabel(track_name, fontsize=22)
    
    # Plot composite case for TCs in IBTrACS only Venn section
    style_subplots(axs[1][0], ib_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[1][1], ib_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[1][2], ib_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[1][3], ib_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[1][4], ib_mean[:, :, 4], vort600_min, vort600_max)
    axs[1][0].set_ylabel("IBTrACS", fontsize=22)
    
    # Plot composite case for TCs in DL only Venn section
    style_subplots(axs[2][0], dl_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[2][1], dl_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[2][2], dl_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[2][3], dl_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[2][4], dl_mean[:, :, 4], vort600_min, vort600_max)
    im = axs[2][0].set_ylabel("Deep Learning Model", fontsize=22)
    
    # Plot composite case for TCs in TRACK and IBTrACS Venn section
    style_subplots(axs[3][0], tr_ib_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[3][1], tr_ib_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[3][2], tr_ib_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[3][3], tr_ib_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[3][4], tr_ib_mean[:, :, 4], vort600_min, vort600_max)
    im = axs[3][0].set_ylabel(track_name + " and IBTrACS", fontsize=22)
    
    # Plot composite case for TCs in TRACK and DL Venn section
    style_subplots(axs[4][0], tr_dl_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[4][1], tr_dl_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[4][2], tr_dl_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[4][3], tr_dl_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[4][4], tr_dl_mean[:, :, 4], vort600_min, vort600_max)
    im = axs[4][0].set_ylabel(track_name + " and DL Model", fontsize=22)
    
    # Plot composite case for TCs in IBTrACS and DL Venn section
    style_subplots(axs[5][0], ib_dl_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[5][1], ib_dl_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[5][2], ib_dl_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[5][3], ib_dl_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[5][4], ib_dl_mean[:, :, 4], vort600_min, vort600_max)
    im = axs[5][0].set_ylabel("IBTrACS and DL Model", fontsize=22)
    
    # Plot composite case for TCs in All Methods Venn section
    style_subplots(axs[6][0], all_methods_mean[:, :, 0], mslp_min, mslp_max)
    style_subplots(axs[6][1], all_methods_mean[:, :, 1], wind_min, wind_max)
    style_subplots(axs[6][2], all_methods_mean[:, :, 2], vort850_min, vort850_max)
    style_subplots(axs[6][3], all_methods_mean[:, :, 3], vort700_min, vort700_max)
    style_subplots(axs[6][4], all_methods_mean[:, :, 4], vort600_min, vort600_max)
    im = axs[6][0].set_ylabel("All Methods", fontsize=22)
    
    # Set Column Labels
    im = axs[0][0].set_title("MSLP", fontsize=22)
    im = axs[0][1].set_title("10m wind speed", fontsize=22)
    im = axs[0][2].set_title("Vorticity at 850hPa", fontsize=22)
    im = axs[0][3].set_title("Vorticity at 700hPa", fontsize=22)
    im = axs[0][4].set_title("Vorticity at 600hPa", fontsize=22)
    
    # Remove ticks
    for row in range(7):
        for col in range(5):
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
    
    for row in range(7):
        for col in range(5):
            axs[row][col].set_aspect(1)
    fig.subplots_adjust(wspace = .35, hspace = -.75)
    fig.tight_layout()
    
    # Save figure
    plt.savefig(save_name + ".pdf")
    
    return mslp_min, mslp_max, wind_min, wind_max, vort850_min, vort850_max, vort700_min, vort700_max, vort600_min, vort600_max, means

def style_subplots(ax, var: np.array, vmin: float, vmax: float, bins: int = 10) -> None:

    '''
    Function to plot means in a subplot and add any styling required
    args:
        ax: axes corresponding to subplot to use
        var: variable to plot (2D array)
        vmin: minimum value for colorbar
        vmax: maximum value for colorbar
        bins: number of levels in colorbar
    returns:
        None
    '''

    im = ax.contourf(var, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, bins))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=12)    

def get_tp_fp_sections():

    '''
    Get a list of sections which have a TC >= Cat1 (tp) and a TC < Cat1 (fp) from IBTrACS
    '''
    
    # Get IBTrACS TC dict having all strgnth TCs
    ibtracs_dict, ibtracs_tracks = get_IBTRACS_dict(cat_opt=1, min_cat=-7)
    
    # Get a dict having the max cat present in a region in a timestep
    dl_cats = {}
    for date in ibtracs_dict.keys():
        date_dict = {}
        for section_id in range(8):
            ib_tcs = get_storms_in_section_by_id(ibtracs_dict[date], section_id)
            if len(ib_tcs) == 0:
                date_dict[section_id] = -9
            else:
                cats = []
                for ib_tc in ib_tcs:
                    _, _, cat = ib_tc
                    cats.append(cat)
                max_cat = np.max(cats)
                date_dict[section_id] = max_cat
        dl_cats[date] = date_dict
    
    # Set up lists
    tp = []
    fp = []
    
    # For each region in each timestep, split into lists accoring to max cat in region
    for date in ibtracs_dict.keys():
        for section_id in range(8):
            if dl_cats[date][section_id] < 1:
                fp.append([date, section_id])
            else:
                tp.append([date, section_id])
    
    return tp, fp

def get_venn_diagram_split_by_cats(ibtracs_dict: dict, track_dict: dict, dl_dict: dict) -> None:

    '''
    Get Venn diagram numbers split by TCs >= Cat1 (tp) and TCs < Cat1 (fp)
    args:
        ibtracs_dict: dict of TCs from IBTrACS
        track_dict: dict of TCs from T-/TRACK
        dl_dict: dict of TCs from TCDetect
    '''
    
    # get split of regions by max category of TC
    tp, fp = get_tp_fp_sections()
    
    all_methods_tp = 0
    all_methods_fp = 0
    tr_ib_tp = 0
    tr_ib_fp = 0
    tr_dl_fp = 0
    tr_dl_tp = 0
    ib_dl_fp = 0
    ib_dl_tp = 0
    tr = 0
    dl_tp = 0
    dl_fp = 0
    ib_tp = 0
    ib_fp = 0
    
    # for each region
    for date in dl_dict.keys():
        for section_id in range(8):
            
            # get TCs in region by each of the three sources
            ib_tcs = get_storms_in_section_by_id(ibtracs_dict[date], section_id)
            dl_tcs = get_storms_in_section_by_id(dl_dict[date], section_id)
            tr_tcs = get_storms_in_section_by_id(track_dict[date], section_id)
            
            # split into right section of Venn diagram
            # then check if region is in the tp or fp section
            
            if len(ib_tcs) > 0 and len(dl_tcs) > 0 and len(tr_tcs) > 0:
                if [date, section_id] in fp:
                    all_methods_fp += 1
                elif [date, section_id] in tp:
                    all_methods_tp += 1
            elif len(ib_tcs) > 0 and len(dl_tcs) > 0 and len(tr_tcs) == 0:
                if [date, section_id] in fp:
                    ib_dl_fp += 1
                elif [date, section_id] in tp:
                    ib_dl_tp += 1
            elif len(ib_tcs) > 0 and len(dl_tcs) == 0 and len(tr_tcs) > 0:
                if [date, section_id] in fp:
                    tr_ib_fp += 1
                elif [date, section_id] in tp:
                    tr_ib_tp += 1
            elif len(ib_tcs) == 0 and len(dl_tcs) > 0 and len(tr_tcs) > 0:
                if [date, section_id] in fp:
                    tr_dl_fp += 1
                elif [date, section_id] in tp:
                    tr_dl_tp += 1
            elif len(ib_tcs) > 0 and len(dl_tcs) == 0 and len(tr_tcs) == 0:
                if [date, section_id] in fp:
                    ib_fp += 1
                elif [date, section_id] in tp:
                    ib_tp += 1
            elif len(ib_tcs) == 0 and len(dl_tcs) > 0 and len(tr_tcs) == 0:
                if [date, section_id] in fp:
                    dl_fp += 1
                elif [date, section_id] in tp:
                    dl_tp += 1
                else:
                    print(date, section_id)
            elif len(ib_tcs) == 0 and len(dl_tcs) == 0 and len(tr_tcs) > 0:
                tr += 1
    
    # print results
    print("\nAll Methods TP:", all_methods_tp)
    print("All Methods FP:", all_methods_fp)
    print("DL-IB TP:", ib_dl_tp)
    print("DL-IB FP:", ib_dl_fp)
    print("DL-TR TP:", tr_dl_tp)
    print("DL-TR FP:", tr_dl_fp)
    print("TR-IB TP:", tr_ib_tp)
    print("TR-IB FP:", tr_ib_fp)
    print("TR:", tr)
    print("DL TP:", dl_tp)
    print("DL FP:", dl_fp)
    print("IB TP:", ib_tp)
    print("IB FP:", ib_fp)
    
def positive_cases_by_cat(model: tf.keras.Model, data: np.array, files: list[str]) -> None:

    '''
    Get number of cases positively and negativly inferred split by max cat TC in the region
    args:
        model: DL model
        data: preprocessed data
        files: list of file paths that correspond to the data given
    '''
    
    # set up counters for positive inference
    no = 0
    minus_5 = 0
    minus_4 = 0
    minus_3 = 0
    minus_2 = 0
    minus_1 = 0
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0

    # set up counters for negative inference
    neg_no = 0
    neg_minus_5 = 0
    neg_minus_4 = 0
    neg_minus_3 = 0
    neg_minus_2 = 0
    neg_minus_1 = 0
    neg_zero = 0
    neg_one = 0
    neg_two = 0
    neg_three = 0
    neg_four = 0
    neg_five = 0
    
    # make inferences
    preds = model.predict(data)
    
    # for each inference, increment relevant counter
    for pred_i, pred in enumerate(preds):
        
        if pred > 0.5:
            
            name = files[pred_i].split("/")[-1]
            cat = name.split("_")[0]
            
            if "no" in cat:
                no += 1
            else:
                cat = int(cat)

                if cat == -5:
                    minus_5 += 1
                elif cat == -4:
                    minus_4 += 1
                elif cat == -3:
                    minus_3 += 1
                elif cat == -2:
                    minus_2 += 1
                elif cat == -1:
                    minus_1 += 1
                elif cat == 0:
                    zero += 1
                elif cat == 1:
                    one += 1
                elif cat == 2:
                    two += 1
                elif cat == 3:
                    three += 1
                elif cat == 4:
                    four += 1
                elif cat == 5:
                    five += 1
                    
        else:
            
            name = files[pred_i].split("/")[-1]
            cat = name.split("_")[0]
            
            if "no" in cat:
                neg_no += 1
            else:
                cat = int(cat)

                if cat == -5:
                    neg_minus_5 += 1
                elif cat == -4:
                    neg_minus_4 += 1
                elif cat == -3:
                    neg_minus_3 += 1
                elif cat == -2:
                    neg_minus_2 += 1
                elif cat == -1:
                    neg_minus_1 += 1
                elif cat == 0:
                    neg_zero += 1
                elif cat == 1:
                    neg_one += 1
                elif cat == 2:
                    neg_two += 1
                elif cat == 3:
                    neg_three += 1
                elif cat == 4:
                    neg_four += 1
                elif cat == 5:
                    neg_five += 1
          
    # collate results
    results = []
    results.append(["No meteorological system", "Unknown", "Post-tropical systems", "Disturbances", "Subtropical systems", "Tropcial Depressions", "Tropcial Storms", "Category 1 TCs", "Category 2 TCs", "Category 3 TCs", "Category 4 TCs", "Category 5 TCs"])
    results.append([no, minus_5, minus_4, minus_3, minus_2, minus_1, zero, one, two, three, four, five])
    results.append([neg_no, neg_minus_5, neg_minus_4, neg_minus_3, neg_minus_2, neg_minus_1, neg_zero, neg_one, neg_two, neg_three, neg_four, neg_five])
    results = pd.DataFrame(results).T
    results.columns = ["IBTrACS Label", "+ve", "-ve"]
    
    # print results
    print(results)
