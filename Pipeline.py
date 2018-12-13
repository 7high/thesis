import csv, glob, itertools, os, shutil, time
import numpy as np
import pandas as pd
import pyshark

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit


class Pipeline():
    
    def __init__(self):
        self.knn = KNeighborsClassifier()
        self.lda = LinearDiscriminantAnalysis()
        self.randomforest = RandomForestClassifier()
    
    def calculate_cm_metrics(self, conf_matrix):
        # Extract the counts
        fp = np.array(conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        fn = np.array(conf_matrix.sum(axis=1) - np.diag(conf_matrix))
        tp = np.diag(conf_matrix)
        tn = np.array(conf_matrix.values.sum() - (fp + fn + tp))
        
        # Put data into dataframe for easy storage
        cm_metrics = pd.DataFrame({'TP':tp, 'FP':fp, 'TN':tn, 'FN':fn}, index=list(conf_matrix.columns))
    
        # Calculate metrics
        cm_metrics["Accuracy"] = (cm_metrics["TP"] + cm_metrics["TN"]) / cm_metrics.iloc[0:3].sum(axis=1)
        cm_metrics["Precision"] = cm_metrics["TP"] / (cm_metrics["TP"] + cm_metrics["FP"])
        cm_metrics["Recall"] = cm_metrics["TP"] / (cm_metrics["TP"] + cm_metrics["FN"])
    
        return cm_metrics
        
    def k_neighbors_classifier(self, X_train, y_train, X_test, y_test):
        time_start = time.time()
        
#        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=2)
        knn_model = self.knn.fit(X_train, y_train)
        
        preds = knn_model.predict(X_test)
        preds_proba = knn_model.predict_proba(X_test)
        score = knn_model.score(X_test, y_test)
                
        time_elapsed = time.time() - time_start
#        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds}
        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds, 'Pred_Proba':preds_proba, 'True':y_test}
    
    def init_srcDir(self):
        src_dir = os.path.dirname(self.SRC_DIR)

        if os.path.exists(src_dir):
            print 'Old ' + src_dir + ' deleted'
            try:
                shutil.rmtree(src_dir)
            except OSError:
                print "Error in deleting " + src_dir
        os.makedirs(src_dir)
    
    def lda_classifier(self, X_train, y_train, X_test, y_test):
        time_start = time.time()
        
#        lda = LinearDiscriminantAnalysis()
        lda_model = self.lda.fit(X_train, y_train)
        
        preds = lda_model.predict(X_test)
        preds_proba = lda_model.predict_proba(X_test)
        score = lda_model.score(X_test, y_test)
          
        time_elapsed = time.time() - time_start
        
        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds, 'Pred_Proba':preds_proba, 'True':y_test}
    
    def make_conf_matrix(self, y_actual, y_pred, cm_type, labels=None):
        """
        Returns a confusion matrix.
        
        Parameters
        ----------
        y_actual: the true values for y
        y_pred: the predicted values for y produced by the classifier
        labels: the labels of y
        
        Returns
        -------
        df_confusion (df): a confusion matrix with the true positive, false positive, true negative and false negative counts
        """
        # Multiclass is mostly used for testing
        if cm_type == 'multiclass':
            # Decode device type from actual and predicted lists
            df_actual = pd.DataFrame(y_actual, columns=labels, dtype=int)
            df_preds = pd.DataFrame(y_pred, columns=labels, dtype=int)
            
        # Binary is used when comparing between only two classes
        # Used in one vs one & one vs all classification schemes
        elif cm_type == 'binary':
            # Create array for the non-class 
            # i.e. if an observation is a positive class, the non-class will be negative
            y_actual_nonclass = np.logical_not(y_actual).astype(int)        
            y_pred_nonclass = np.logical_not(y_pred).astype(int)
            
            # Combine positive and nonclass into dfs
            if labels != None:
                # Use labels
                df_actual = pd.DataFrame({labels[0]: y_actual, labels[1]:y_actual_nonclass})        
                df_preds = pd.DataFrame({labels[0]: y_pred, labels[1]:y_pred_nonclass})
            else:
                # Use "rest" label
                df_actual = pd.DataFrame({y_actual.name: y_actual, 'rest': y_actual_nonclass})        
                df_preds = pd.DataFrame({y_actual.name: y_pred, 'rest':y_pred_nonclass})

        else:
            print "Invalid confusion matrix type"
            return
        
        # Create column containing the response variable
        df_actual["Response"] = (df_actual.iloc[:] == 1).idxmax(1)
        df_preds["Response"] = (df_preds.iloc[:] == 1).idxmax(1)  
        
        # Put into numpy array for easy manipulation into crosstab
        actual = np.array(df_actual["Response"])
        pred = np.array(df_preds["Response"])        
        
        # Create crosstab
        df_confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])

        # Fix non-square confusion matrices
        if len(df_confusion.columns) != len(df_confusion.index):
            # Get set of column and index entries
            columns = set(df_confusion.columns)
            rows = set(df_confusion.index)

            # Find which entries are missing from both columns and indices
            missing_columns = rows - columns
            missing_rows = columns - rows

            # Create array of padded zeros
            len_longSide = max(len(columns), len(rows))
            zeros = np.zeros((len_longSide,1), dtype=np.int)    

            # Fill missing column/rows
            for column in missing_columns:
                df_confusion[column] = zeros
            for row in missing_rows:
                # For rows, transpose df first to allow easy concat
                temp_cm = df_confusion.transpose()
                temp_cm[row] = zeros
                df_confusion = temp_cm.transpose()            
        
        
        return df_confusion
    
    def printout_cm_metrics(self, device_type, cm_list, matrices, metrics):
        print "Device Type:", device_type
        print "=========================="
        
        for i, cm in enumerate(cm_list):
            print cm_list[i]
            print ""
            print "Confusion Matrix"
            print matrices[i]
            print ""
            print "Metrics"
            print metrics[i]
            print "---------------------------------------------------------"
            
    def one_vs_all_classify(self, df, features_list, y_list, output='print'):
        time_start = time.time()
        
        onevsall_dict = {}
        
        # Divide df by train and test devices
        df_test = df[df["Set"]=="test"]
        df_train = df[df["Set"]=="train"]
        
        # Train using chosen features
        X_train = df_train[features_list]
        X_test = df_test[features_list]
    
        for device_type in y_list:
            # Set one device type as y
            y_train = df_train[device_type]
            y_test = df_test[device_type]
    
            time_start_clf = time.time()
    
            rf_clf = self.random_forest_classifier(X_train, y_train, X_test, y_test)
            knn_clf = self.k_neighbors_classifier(X_train, y_train, X_test, y_test)
            lda_clf = self.lda_classifier(X_train, y_train, X_test, y_test)
    
            time_elapsed_clf = time.time() - time_start_clf

            # Get confusion matrices
            rf_cm = self.make_conf_matrix(y_test, rf_clf['Pred'], cm_type='binary')
            knn_cm = self.make_conf_matrix(y_test, knn_clf['Pred'], cm_type='binary')
            lda_cm = self.make_conf_matrix(y_test, lda_clf['Pred'], cm_type='binary')
            
            # Calculate metrics
            rf_metrics = self.calculate_cm_metrics(rf_cm)
            knn_metrics = self.calculate_cm_metrics(knn_cm)
            lda_metrics = self.calculate_cm_metrics(lda_cm)
            
#            print "Random Forest Score:", rf_clf['Score'], "Time: ", rf_clf['Time']
#            print "KNN Score:", knn_clf['Score'], "Time: ", knn_clf['Time']
#            print "LDA Score:", lda_clf['Score'], "Time: ", lda_clf['Time']
            
            if output == 'print':
                self.printout_cm_metrics(device_type,['RF','KNN','LDA'],[rf_cm, knn_cm, lda_cm],[rf_metrics, knn_metrics, lda_metrics])
                #Print outs
#                print "Device Type:", device_type
#                print "--------------------------"
#                print "--------------------------"
#                print "RF Confusion Matrix\n", rf_cm, '\n'   
#                print "RF Metrics\n", rf_metrics, '\n'      
#                print "--------------------------"
#                print "KNN Confusion Matrix\n", knn_cm, '\n'
#                print "KNN Metrics\n", knn_metrics, '\n'
#                print "--------------------------"
#                print "LDA Confusion Matrix\n", lda_cm, '\n'
#                print "LDA Metrics\n", lda_metrics, '\n'
#                print "--------------------------"
                
                print "Total time (classifiers):", time_elapsed_clf
                print ""
            elif output == 'file':
                print 'file output'
                #TODO: Add option to output to file
            
            onevsall_dict[device_type] = {'RF': {'Classifier': rf_clf, 'CM': rf_cm, 'Metrics':rf_metrics},
                                          'KNN': {'Classifier': knn_clf, 'CM': knn_cm, 'Metrics':knn_metrics},
                                          'LDA': {'Classifier': lda_clf, 'CM': lda_cm, 'Metrics':lda_metrics}}
        
        total_time = time.time() - time_start
        print "Total time (one vs all_classify):", total_time
        print ""    
        
        return onevsall_dict, total_time
        
        
    def one_vs_one_classify(self, df, features_list, y_list):
        time_start = time.time()
        
        onevsone_dict = {}
        
        
        # Get possible combinations for one vs one
        combinations = [combination for combination in itertools.combinations(y_list, 2)]

        for device_pair in combinations:
            # Only use data with the two device types needed for one vs one classification
            pos_device_type = device_pair[0]
            neg_device_type = device_pair[1]
            df_1v1 = df[(df["DeviceType"]==pos_device_type) | (df["DeviceType"]==neg_device_type)]
    
            # Separate df into train and test sets
            df_train = df_1v1[df_1v1["Set"]=="train"]
            df_test = df_1v1[df_1v1["Set"]=="test"]
            X_train = df_train[features_list]
            X_test = df_test[features_list]
            y_train = df_train[pos_device_type]
            y_test = df_test[pos_device_type]
            
            time_start_clf = time.time()
            
            # Run classifiers
            rf_clf = self.random_forest_classifier(X_train, y_train, X_test, y_test)
            knn_clf = self.k_neighbors_classifier(X_train, y_train, X_test, y_test)
            lda_clf = self.lda_classifier(X_train, y_train, X_test, y_test)
    
            time_elapsed_clf = time.time() - time_start_clf
    
            
            # Get confusion matrices
            rf_cm = self.make_conf_matrix(y_test, rf_clf['Pred'], cm_type='binary', labels=device_pair)
            knn_cm = self.make_conf_matrix(y_test, knn_clf['Pred'], cm_type='binary', labels=device_pair)
            lda_cm = self.make_conf_matrix(y_test, lda_clf['Pred'], cm_type='binary', labels=device_pair)
            
            # Calculate metrics
            rf_metrics = self.calculate_cm_metrics(rf_cm)
            knn_metrics = self.calculate_cm_metrics(knn_cm)
            lda_metrics = self.calculate_cm_metrics(lda_cm)
            
            #Print outs
            print "Device Pair:", device_pair
            print "--------------------------"
            print "--------------------------"
            print "RF Confusion Matrix\n", rf_cm, '\n'   
            print "RF Metrics\n", rf_metrics, '\n'      
            print "--------------------------"
            print "KNN Confusion Matrix\n", knn_cm, '\n'
            print "KNN Metrics\n", knn_metrics, '\n'
            print "--------------------------"
            print "LDA Confusion Matrix\n", lda_cm, '\n'
            print "LDA Metrics\n", lda_metrics, '\n'
            print "--------------------------"
            
            print "Total time (classifiers):", time_elapsed_clf
            print ""
            
            # Store all into dictionary and list
            onevsone_dict[device_pair] = {'RF': {'Classifier': rf_clf, 'CM': rf_cm, 'Metrics':rf_metrics},
                                          'KNN': {'Classifier': knn_clf, 'CM': knn_cm, 'Metrics':knn_metrics},
                                          'LDA': {'Classifier': lda_clf, 'CM': lda_cm, 'Metrics':lda_metrics}}
        
        total_time = time.time() - time_start
        print "Total time (one vs one_classify):", total_time
        print ""
        
        return onevsone_dict, total_time
        

    def random_forest_classifier(self, X_train, y_train, X_test, y_test):
        time_start = time.time()
        
    #     randomforest = RandomForestClassifier(random_state=0, n_jobs=2)
#        randomforest = RandomForestClassifier(random_state=0, n_jobs=2, class_weight='balanced')
        rf_model = self.randomforest.fit(X_train, y_train)
    
        preds = rf_model.predict(X_test)
        preds_proba = rf_model.predict_proba(X_test)
        score = rf_model.score(X_test, y_test)
               
        time_elapsed = time.time() - time_start
        
        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds, 'Pred_Proba':preds_proba, 'True':y_test}
#        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds, 'Pred_Proba':preds_proba}
#        return {'Score' : score, 'Time' : time_elapsed, 'Pred': preds}
    
    def resample(self, df, kind, category):
        # Based on the kind of resampling, get the majority, minority or median class count
        if kind == "under":
            count = min(df[category].value_counts())
        elif kind == "over":
            count = max(df[category].value_counts())
        elif kind == "mid":
            count = int(np.median(df[category].value_counts()))
        else:
            print "Invalid resampling"
            return
        
        # Get all unique values in the given category
        uniques = df[category].unique()
        
        # Resample all sub_dfs based on the unique values of the category
        sub_dfs = []
        for value in uniques:
            if kind == "under":
                sub_df = df[df[category] == value].sample(count)
            elif kind == "over" or kind == "mid":
                sub_df = df[df[category] == value].sample(count, replace=True)
            sub_dfs.append(sub_df)
            
        # Recombine sampled sub_dfs into one sampled df
        comb_sub_dfs = pd.concat(sub_dfs, axis=0)
        return comb_sub_dfs
    
#------------------------------------------------------------------------------------------------------------
class BLEPipeline(Pipeline):
    # Global Variables    
    devices_devicenames = ['August1', 'August2', 'Door1', 'Door2', 
                           'Kevo', 'Push', 'Room1', 'Room2', 'Weather']
    devices_publicaddrs = ['Home1', 'Home2']
    
    id_devicenames = [['Kevo','Unikey'],
                    'Eve Door 91B3',
                    'Eve Door DC42',
                    's',
                    'Aug',
                    'L402EL4',
                    'Eve Weather 943D',
                    'Eve Room 8F24',
                    'Eve Room 4A04']
    
    BLE_DEVICES = sorted(devices_devicenames + devices_publicaddrs)
    
    # Devices that can be identified using public (static) advertising addresses
    DEVICES_PUBLICADDRS = {'ec:fe:7e:14:44:be' : 'Home1', 
                           'ec:fe:7e:14:44:a1' : 'Home2'}
    
    # Devices that can be identified using device names
    DEVICES_NAMES = {'August1': 'L402EL4',
                    'August2': 'Aug',
                    'Door1': 'Eve Door 91B3',
                    'Door2': 'Eve Door DC42',
                    'Kevo': ['Kevo', 'Unikey'],
                    'Push': 's',
                    'Room1': 'Eve Room 4A04',
                    'Room2': 'Eve Room 8F24',
                    'Weather': 'Eve Weather 943D'}
    
    # Just the reverse of DEVICES_NAMES
    NAMES_DEVICES = {'Aug': 'August2',
                     'Eve Door 91B3': 'Door1',
                     'Eve Door DC42': 'Door2',
                     'Eve Room 4A04': 'Room1',
                     'Eve Room 8F24': 'Room2',
                     'Eve Weather 943D': 'Weather',
                     'Kevo': 'Kevo',
                     'L402EL4': 'August1',
                     'Unikey': 'Kevo',
                     's': 'Push'}
    
    DEVICE_TYPE = {'August1': 'lock',
                   'August2': 'lock',
                   'Door1': 'door',
                   'Door2': 'door',
                   'Home1': 'door',
                   'Home2': 'door',
                   'Kevo': 'lock',
                   'Push': 'temp',
                   'Room1': 'temp',
                   'Room2': 'temp',
                   'Weather': 'temp'}
    
    TRAINING_TEST = {'August1': 'train',
                     'August2': 'test',
                     'Door1': 'train',
                     'Door2': 'test',
                     'Home1': 'train',
                     'Home2': 'train',
                     'Kevo': 'train',
                     'Push': 'train',
                     'Room1': 'train',
                     'Room2': 'test',
                     'Weather': 'train'}
    
    PDU_TYPES = {0: 'ADV_IND',
                 1: 'ADV_DIRECT_IND',
                 2: 'ADV_NONCONN_IND',
                 3: 'SCAN_REQ',
                 4: 'SCAN_RSP',
                 5: 'CONNECT_REQ',
                 6: 'ADV_SCAN_IND'}
    
    SRC_DIR = './BLE_Source/'
    DST_DIR = './BLE_Destination/'
    PCAP_DIR = '/root/Documents/Thesis/BLE_PCAPS/'
    TIMING_PKT_NUMBER = 25000
    
    FEATURES = ['Name', 'DeviceName', 'AccessAddr', 'AdvertAddr', 'BLE_LL_Length', 
                'PDUTypeNum', 'TxAddr', 'CompanyID','ScanAddr',
                'RFChannel', 'PacketLength', 'Time']
    
    path_name = os.getcwd()
    DATE = path_name[path_name.rindex('/')+1:]
    PROC_TIME = "ble_processing_time_" + DATE + ".csv"
    
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.lda = LinearDiscriminantAnalysis(n_components=1, solver='lsqr')
        self.randomforest = RandomForestClassifier(max_features=14)
    
    def count_assoc_pkts(self, df, device):
        """
        Gets the count of packets of a given device that are sent within a second of each other (associated packets)
        
        Parameters
        ----------
        df: (dataframe) the dataframe containing the packet information
        device: (string) the name of the device for which the assoc_pkt count will be calculated
        
        Output
        ------
        None
        
        Returns
        -------
        assoc_count: (pandas series) contains the assoc_packet count for each packet. 
                    Uses the index of the packet from the dataframe
        """
            
        ASSOC_PKT_THRESHOLD = 1 # the threshold in seconds within which a packet will be considered an assoc_pkt
    
        # Extract time values of all packets belonging to a certain device
        df_device = df[df["Name"]==device]
        pkt_time_values = np.array(df_device["Time"].values)
        
        assoc_pkt_counts = []
        
        # Iterate through each packet of the device
        for pkt_index in range(0,len(df_device)):  
    
            # Create an array of size=len(pkt_time_values) that contains the time value of packet X
            pkt_time = np.full((len(pkt_time_values),),df_device.iloc[pkt_index]["Time"])
    
            # Calculate the time difference between packet X and all other packets
            diff = np.abs(np.subtract(pkt_time, pkt_time_values))
    
            # Calculate the count of packets that would be considered an assoc_pkt based on ASSOC_PKT_THRESHOLD
            assoc_pkts = (diff <= ASSOC_PKT_THRESHOLD).sum()
            assoc_pkt_counts.append(assoc_pkts)
            
        
        assoc_count = pd.Series(assoc_pkt_counts, index=df_device.index)
        return assoc_count

    def extract_packet_features(self, filename = os.path.join(PCAP_DIR, 'master.pcap'), create_master=True, printout=False):
        """
        Unit that extracts wanted features out of packets in a packet capture file.
        The feature_extractor focuses on features derived from packet information. 
        Secondary features are processed by the make_dataframe function.
        Produces two csv files for each device in WIFI_DEVICES (see Global Variables).
        One file is for all packets where the device is the source; the other is where the device is the destination.
        
        Parameters
        ----------
        filename: (string) the absolute path of the packet capture file
        
        Output
        ------
        Source directory: (filesystem) creates a directory containing csv files for each device 
            where it is the source of the packet
        Destination directory: (filesystem) creates a directory containing csv files for each device 
            where it is the destination of the packet
        
        Returns
        -------
        none
        """
        
        # Prepare writers
        pt_file = open(self.PROC_TIME, 'w')
        csv.writer(pt_file).writerow(["Unit", "Total Packets Processed", "Total Process Time", "Average Process Time"])
        pt_file.close()
    
        # Initialize counters
        pkt_count = 0
        total_time_processing = 0
        total_time_start = time.time()
    
        # Initialize dicts for each device
        tgt_files_by_src = {}
        
        # Combine all pcaps in directory in one master pcap
        if (create_master):
            try:
                if os.path.exists("/root/Documents/Thesis/BLE_PCAPS/master.pcap"):
                    os.remove("/root/Documents/Thesis/BLE_PCAPS/master.pcap")
                    
                ret = os.system('mergecap /root/Documents/Thesis/BLE_PCAPS/*.pcap -w /root/Documents/Thesis/BLE_PCAPS/master.pcap')
                if ret != 0:
                    raise OSError
            except OSError:
                print 'Could not make master capture file'
    
        # Initialize capture file 
        cap = pyshark.FileCapture(filename, only_summaries=False)
    
        # Get time of first packet
        prev_pkt_time = cap[0].frame_info.time_epoch
    
        # Initialize output folders
        self.init_srcDir()
        
        # Open output files for each Wi-Fi device
        for device in self.BLE_DEVICES:
            tgt_files_by_src[device] = open(self.SRC_DIR + device + ".csv", 'a')
            
            # Initialize with column headers
            csv.writer(tgt_files_by_src[device]).writerow(self.FEATURES)
        
        # Go through each packet in capture, and store pertinent packets to csv files
        for pkt in cap:
            if printout:
                if pkt_count % self.TIMING_PKT_NUMBER == 0:
                    print "Working packet #", pkt_count, "..."
            pkt_count += 1
    
            time_start_singlepacket = time.time()
            self.parse_packet(pkt, tgt_files_by_src)
            total_time_processing += time.time() - time_start_singlepacket
    
        total_time_elapsed = time.time() - total_time_start
        
        # Close files
        for open_file in tgt_files_by_src.values():
            open_file.close()
            
        # Calculate time variables
        final_time = time.time()
        normalized_total_time = (self.TIMING_PKT_NUMBER * total_time_elapsed) / pkt_count
        normalized_processing_time = (self.TIMING_PKT_NUMBER * total_time_processing) / pkt_count
    
        # Print time variables
        print "Total number of packets processed: ", pkt_count
        print "Total data processing time: ", total_time_elapsed
        print "Normalized total processing time per 25k packets: ", normalized_total_time
        print "Total capture file processing time: ", total_time_processing
        print "Normalized capture file processing time: ", normalized_processing_time
    
        # Print out time metrics to csv
        pt_file = open(self.PROC_TIME, 'a')
        csv.writer(pt_file).writerow(["Packet capture iteration", pkt_count, 
                                      total_time_processing, normalized_processing_time])
        csv.writer(pt_file).writerow(["Component start and finish time", total_time_start, 
                                      final_time, final_time-total_time_start])
        pt_file.close()

  

    def make_dataframe(self, path='/root/Documents/Thesis/Code/BLE_Source'):
        """
        Unit that takes all the csv files produced by the feature_extractor unit 
        and puts them into a pandas dataframe.
        Returns a clean dataframe with all good data
    
        Parameters
        ----------
        path: (filesystem) the absolute path of the folder containing the csv files
    
        Output
        ------
        none
    
        Returns
        -------
        dataframe: (pandas dataframe) a useful data structure for machine learning
        counts: (pandas series) packet counts for each device 
        """
        
        # Search the path for csv files
        all_csvs = glob.glob(os.path.join(path, "*.csv"))
    
        # Collect all csvs in one dataframe
        df_from_each_file = (pd.read_csv(f) for f in all_csvs)
        df = pd.concat(df_from_each_file, ignore_index=True, sort=False)
    
        # Add device type of each packet
        df["DeviceType"] = df["Name"].map(self.DEVICE_TYPE)
            
        # Add whether device is a training or test device
        df["Set"] = df["Name"].map(self.TRAINING_TEST)
        
        # One-hot encode device type (response variable)
        deviceType_series = pd.get_dummies(df["DeviceType"])
        df = pd.concat([df, deviceType_series], axis=1)
        
        # TODO: One-hot encode company ID 
        
        # TODO: One-hot encode access address
        
        # TODO: One-hot encode adv address
        
        # TODO: One-hot encode scanning address
        
        # One-hot encode PDU_type
        df["PDUType"] = df["PDUTypeNum"].map(self.PDU_TYPES)
        pduType_series = pd.get_dummies(df["PDUType"])
        df = pd.concat([df, pduType_series], axis=1)
        
        # Get number of associated packets for each packet
        list_assoc_pkts = []
    #     for device in list(df["Name"].unique()):
        for device in self.BLE_DEVICES:
            assoc_pkts = self.count_assoc_pkts(df, device)
            list_assoc_pkts.append(assoc_pkts)
        df["Assoc_Packets"] = pd.concat(list_assoc_pkts)
        
        # Fill NaNs with 0
        df["CompanyID"] = df["CompanyID"].fillna(0)
        df["ScanAddr"] = df["ScanAddr"].fillna(0)
        
        # Count packets for each device
        device_counts = df["Name"].value_counts()
        print device_counts
            
        return df


    def parse_packet(self, pkt, tgt_files_by_src):
        """
        Parses a given packet and extracts the following features:
            (BLE LL)
            - device name
            - access address
            - advertising address
            - BLE LL packet length (bytes)
            - PDU type
            - Tx address type (public or random)
            - company id
            - scanning address (if SCAN_REQ pdu_type)
            
            (BLE RF)
            - rf channel (same as advertising channel: RF 0 = ADV 37, 12 = 38, 39 = 39)
            
            (Frame)
            - total frame length (bytes)
            - epoch time (seconds)      
            
        The features of the packet are written out to a csv row, which is
        in turn written out to a csv file in the given dictionaries.
        
        Parameters
        ----------
        pkt: (Pyshark packet object) the packet from which features will be extracted
        tgt_files_by_src: (dictionary) a dictionary of open csv files.
            The keys are device source addresses, and the values are the open csv files.
        tgt_files_by_dst: (dictionary) a dictionary of open csv files.
            The keys are device destination addresses, and the values are the open csv files.
        """
        
        public_addrs = self.DEVICES_PUBLICADDRS.keys()
        known_names = self.NAMES_DEVICES.keys()
        
        try:        
            # Find devices with known advertising addresses or device_names
            advAddr = pkt.btle.get_field_value('advertising_address')        
            name = pkt.btle.get_field_value('btcommon_eir_ad_entry_device_name')
            
            # Assign an identifier based on whether a known advAddr or name was found
            identifier, identifier_type = (advAddr,'advAddr') if name == None else (name,'name')       
            
            if (identifier in public_addrs) or (identifier in known_names):
                           
                # BLE LL features
                deviceName = pkt.btle.get_field_value('btcommon_eir_ad_entry_device_name')
                accessAddr = pkt.btle.get_field_value('access_address')
                advAddr = pkt.btle.get_field_value('advertising_address')
                bleLength = pkt.btle.get_field_value('length')
                pduType = pkt.btle.get_field_value('advertising_header_pdu_type')
                scanAddr = pkt.btle.get_field_value('scanning_address')
                
                # BLE RF
                rfChannel = pkt.btle_rf.get_field_value('channel')
                
                # Bluetooth
                pktLength = pkt.frame_info.get_field_value('len')
                epochTime = pkt.frame_info.get_field_value('time_epoch')       
                
                # Name as used in thesis document
                name = self.DEVICES_PUBLICADDRS[identifier] if identifier_type == 'advAddr' else self.NAMES_DEVICES[identifier]
                            
                # Output matches the order of FEATURES
                output = [name, deviceName, accessAddr, advAddr, bleLength, pduType, txAddr, companyID, scanAddr,
                          rfChannel,
                          pktLength, epochTime]
                
                # Write features to csv           
                csv.writer(tgt_files_by_src[name]).writerow(output)
               
        
        except AttributeError:
            print "ignored: ", pkt.number            



#------------------------------------------------------------------------------------------------------------
class WifiPipeline(Pipeline): 
    
    # Global Variables
    ROUTER = '78:d2:94:4d:ab:3e'
    WIFI_DEVICES = ['ec:1a:59:e4:fd:41', 'ec:1a:59:e4:fa:09',
                    'ec:1a:59:e5:02:0d', '14:91:82:24:dd:35',
                    '60:38:e0:ee:7c:e5', '14:91:82:cd:df:3d',
                    'b4:75:0e:0d:94:65', 'b4:75:0e:0d:33:d5',
                    '94:10:3e:2b:7a:55', '30:8c:fb:3a:1a:ad',
                    'd0:73:d5:26:b8:4c', 'd0:73:d5:26:c9:27',
                    'ac:84:c6:97:7c:cc', 'b0:4e:26:c5:2a:41',
                    '70:4f:57:f9:e1:b8', ROUTER]

    DEVICE_NAME = {'ec:1a:59:e4:fd:41' : 'Netcam1', 
                   'ec:1a:59:e4:fa:09' : 'Netcam2',
                   'ec:1a:59:e5:02:0d' : 'Netcam3',
                   '14:91:82:24:dd:35' : 'Insight',
                   '60:38:e0:ee:7c:e5' : 'Mini',
                   '14:91:82:cd:df:3d' : 'Switch1',
                   'b4:75:0e:0d:94:65' : 'Switch2',
                   'b4:75:0e:0d:33:d5' : 'Switch3',
                   '94:10:3e:2b:7a:55' : 'Switch4',
                   '30:8c:fb:3a:1a:ad' : 'Dropcam',
                   'd0:73:d5:26:b8:4c' : 'Lifx1', 
                   'd0:73:d5:26:c9:27' : 'Lifx2',
                   'ac:84:c6:97:7c:cc' : 'Kasa', 
                   'b0:4e:26:c5:2a:41' : 'TpBulb',
                   '70:4f:57:f9:e1:b8' : 'TpPlug',
                    ROUTER : 'Router'}

    DEVICE_TYPE = {'ec:1a:59:e4:fd:41' : 'camera',
                   'ec:1a:59:e4:fa:09' : 'camera',
                   'ec:1a:59:e5:02:0d' : 'camera',
                   '14:91:82:24:dd:35' : 'plug',
                   '60:38:e0:ee:7c:e5' : 'plug',
                   '14:91:82:cd:df:3d' : 'plug',
                   'b4:75:0e:0d:94:65' : 'plug',
                   'b4:75:0e:0d:33:d5' : 'plug',
                   '94:10:3e:2b:7a:55' : 'plug',
                   '30:8c:fb:3a:1a:ad' : 'camera',
                   'd0:73:d5:26:b8:4c' : 'bulb', 
                   'd0:73:d5:26:c9:27' : 'bulb',
                   'ac:84:c6:97:7c:cc' : 'camera', 
                   'b0:4e:26:c5:2a:41' : 'bulb',
                   '70:4f:57:f9:e1:b8' : 'plug',
                    ROUTER : 'router'}

    TRAINING_TEST = {'ec:1a:59:e4:fd:41' : 'train', 
                     'ec:1a:59:e4:fa:09' : 'train',
                     'ec:1a:59:e5:02:0d' : 'test',
                     '14:91:82:24:dd:35' : 'train',
                     '60:38:e0:ee:7c:e5' : 'train',
                     '14:91:82:cd:df:3d' : 'train',
                     'b4:75:0e:0d:94:65' : 'train',
                     'b4:75:0e:0d:33:d5' : 'train',
                     '94:10:3e:2b:7a:55' : 'test',
                     '30:8c:fb:3a:1a:ad' : 'train',
                     'd0:73:d5:26:b8:4c' : 'train', 
                     'd0:73:d5:26:c9:27' : 'test',
                     'ac:84:c6:97:7c:cc' : 'test', 
                     'b0:4e:26:c5:2a:41' : 'train',
                     '70:4f:57:f9:e1:b8' : 'test'}

    DATA_PKT_SUBTYPES = {32 : 'Data',
                         40 : 'QoS_Data',
                         44 : 'QoS_Null'}

    FEATURES = ["Time", "PacketLength", "Duration", 
                "SourceAddr", "DestAddr", "SubtypeNum"]
    
    SRC_DIR = './Wifi_Source/'
    DST_DIR = './Wifi_Destination/'
    PCAP_DIR = '/root/Documents/Thesis/PCAPS'
    TIMING_PKT_NUMBER = 25000
    DATA_FRAME_TYPE = '2'

    path_name = os.getcwd()
    DATE = path_name[path_name.rindex('/')+1:]
    PROC_TIME = "wifi_processing_time_" + DATE + ".csv"
    
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=9)
        self.lda = LinearDiscriminantAnalysis(n_components=1, solver='lsqr')
        self.randomforest = RandomForestClassifier(max_features=8)
    
    def count_assoc_pkts(self, df, device):
        """
        Gets the count of packets of a given device that are sent 
        within a second of each other (associated packets)
        
        Parameters
        ----------
        df: (dataframe) the dataframe containing the packet information
        device: (string) the name of the device for which the assoc_pkt count will be calculated
        
        Output
        ------
        None
        
        Returns
        -------
        assoc_count: (pandas series) contains the assoc_packet count for each packet. 
                    Uses the index of the packet from the dataframe
        """
            
        ASSOC_PKT_THRESHOLD = 1 # the threshold in seconds within which a packet will be considered an assoc_pkt
    
        # Extract time values of all packets belonging to a certain device
        df_device = df[df["Name"]==device]
        pkt_time_values = np.array(df_device["Time"].values)
        
        assoc_pkt_counts = []
        
        # Iterate through each packet of the device
        for pkt_index in range(0,len(df_device)):  
    
            # Create an array of size=len(pkt_time_values) that contains the time value of packet X
            pkt_time = np.full((len(pkt_time_values),),df_device.iloc[pkt_index]["Time"])
    
            # Calculate the time difference between packet X and all other packets
            diff = np.abs(np.subtract(pkt_time, pkt_time_values))
    
            # Calculate the count of packets that would be considered an assoc_pkt based on ASSOC_PKT_THRESHOLD
            assoc_pkts = (diff <= ASSOC_PKT_THRESHOLD).sum()
            assoc_pkt_counts.append(assoc_pkts)
            
        
        assoc_count = pd.Series(assoc_pkt_counts, index=df_device.index)
        return assoc_count
    
    def extract_packet_features(self, filename = os.path.join(PCAP_DIR, 'master.cap'), create_master=True, printout=False):
        """
        Unit that extracts wanted features out of packets in a packet capture file.
        The feature_extractor focuses on features derived from packet information. 
        Secondary features are processed by the make_dataframe function.
        Produces two csv files for each device in WIFI_DEVICES (see Global Variables).
        One file is for all packets where the device is the source; 
        the other is where the device is the destination.

        Parameters
        ----------
        filename: (string) the absolute path of the packet capture file

        Output
        ------
        Source directory: (filesystem) creates a directory containing 
            csv files for each device where it is the source of the packet
        Destination directory: (filesystem) creates a directory containing 
            csv files for each device where it is the destination of the packet

        Returns
        -------
        none
        """

        # Prepare writers
        pt_file = open(self.PROC_TIME, 'w')
        csv.writer(pt_file).writerow(["Unit", "Total Packets Processed", 
                                      "Total Process Time", "Average Process Time"])
        pt_file.close()

        # Initialize counters
        pkt_count = 0
        total_time_processing = 0
        total_time_start = time.time()

        # Initialize dicts for each device
        tgt_files_by_src = {}

        # Combine all pcaps in directory in one master pcap
        if (create_master):
            try:
                if os.path.exists("/root/Documents/Thesis/PCAPS/master.cap"):
                    os.remove("/root/Documents/Thesis/PCAPS/master.cap")

                ret = os.system('mergecap /root/Documents/Thesis/PCAPS/wifi* -w /root/Documents/Thesis/PCAPS/master.cap')
                if ret != 0:
                    raise OSError
            except OSError:
                print 'Could not make master capture file'

        # Initialize capture file 
        cap = pyshark.FileCapture(filename, only_summaries=False)

        # Get time of first packet
        prev_pkt_time = cap[0].frame_info.time_epoch

        # Initialize output folders
        self.init_srcDir()

        # Open output files for each Wi-Fi device
        for device in self.WIFI_DEVICES:
            tgt_files_by_src[device] = open(self.SRC_DIR + device.replace(':', '.') + ".csv", 'a')

            # Initialize with column headers
            csv.writer(tgt_files_by_src[device]).writerow(self.FEATURES)

        # Go through each packet in capture, and store pertinent packets to csv files
        for pkt in cap:
            if printout:
                if pkt_count % self.TIMING_PKT_NUMBER == 0:
                    print "Working packet #", pkt_count, "..."
            pkt_count += 1

            time_start_singlepacket = time.time()
            if pkt.wlan.fc_type == self.DATA_FRAME_TYPE:
                self.parse_packet(pkt, tgt_files_by_src)
                total_time_processing += time.time() - time_start_singlepacket

        total_time_elapsed = time.time() - total_time_start

        # Close files
        for open_file in tgt_files_by_src.values():
            open_file.close()

        # Rename files to device names for readability
        self.rename_csv_files(self.DEVICE_NAME)

        # Calculate time variables
        final_time = time.time()
        normalized_total_time = (self.TIMING_PKT_NUMBER * total_time_elapsed) / pkt_count
        normalized_processing_time = (self.TIMING_PKT_NUMBER * total_time_processing) / pkt_count

        # Print time variables
        print "Total number of packets processed: ", pkt_count
        print "Total data processing time: ", total_time_elapsed
        print "Normalized total processing time per 25k packets: ", normalized_total_time
        print "Total capture file processing time: ", total_time_processing
        print "Normalized capture file processing time: ", normalized_processing_time

        # Print out time metrics to csv
        pt_file = open(self.PROC_TIME, 'a')
        csv.writer(pt_file).writerow(["Packet capture iteration", pkt_count, 
                                      total_time_processing, normalized_processing_time])
        csv.writer(pt_file).writerow(["Component start and finish time", 
                                      total_time_start, final_time, final_time-total_time_start])
        pt_file.close()
       

    def get_mac_vendors(self):
        """
        Uses the macvendors.co API to lookup the vendors of Wi-Fi devices.
        Requires internet access.
        
        Parameters
        ----------
        None
        
        Output
        ------
        None
        
        Returns
        -------
        device_vendors (dict): keys(str) = WIFI_DEVICES MAC addresses, 
                               values(str) = vendor names
        """
        import json, requests
    
        # Get JSON response from API
        vendors_json = []
        for addr in self.WIFI_DEVICES:
            response = requests.get('http://macvendors.co/api/' + addr).text
            vendors_json.append(response)
    
        # Extracting company from API response
        vendors = []
        for vendor_json in vendors_json:
            response = json.loads(vendor_json)
            company = str(response['result']['company']).split(' ',1)[0].capitalize()
            vendors.append(company)
    
        # Put device MAC addresses and vendors into dictionary
        device_vendors = dict(zip(self.WIFI_DEVICES, vendors))
        
        return device_vendors            
            
    def make_dataframe(self, path='/root/Documents/Thesis/Code/Wifi_Source'):
        """
        Unit that takes all the csv files produced by the 
        feature_extractor unit and puts them into a pandas dataframe.
        Returns a clean dataframe with all good data
    
        Parameters
        ----------
        path: (filesystem) the absolute path of the folder containing the csv files
    
        Output
        ------
        none
    
        Returns
        -------
        dataframe: (pandas dataframe) a useful data structure for machine learning
        counts: (pandas series) packet counts for each device 
        """
        
        # Search the path for csv files
        all_csvs = glob.glob(os.path.join(path, "*.csv"))
    
        # Collect all csvs in one dataframe
        df_from_each_file = (pd.read_csv(f) for f in all_csvs)
        df = pd.concat(df_from_each_file, ignore_index=True, sort=False)
    
        # Add device type, device ID of each packet
        df["DeviceType"] = df["SourceAddr"].map(self.DEVICE_TYPE)
        df["Name"] = df["SourceAddr"].map(self.DEVICE_NAME)
        
        # Add whether device is a training or test device
        df["Set"] = df["SourceAddr"].map(self.TRAINING_TEST)
        
        # One-hot encode device type (response variable)
        deviceType_series = pd.get_dummies(df["DeviceType"])
        df = pd.concat([df, deviceType_series], axis=1)
        
        # One-hot encode MAC vendors
        df["Vendor"] = df["SourceAddr"].map(self.get_mac_vendors())
        vendor_series = pd.get_dummies(df["Vendor"])
        df = pd.concat([df, vendor_series], axis=1)
    
        # One-hot encode packet subtype
        df["Subtype"] = df["SubtypeNum"].map(self.DATA_PKT_SUBTYPES)
        subtype_series = pd.get_dummies(df["Subtype"])
        df = pd.concat([df, subtype_series], axis=1)   
        
        # Get number of associated packets for each packet
        list_assoc_pkts = []
        
        for device in self.DEVICE_NAME.values():
            assoc_pkts = self.count_assoc_pkts(df, device)
            list_assoc_pkts.append(assoc_pkts)
        df["Assoc_Packets"] = pd.concat(list_assoc_pkts)
        
        # Count packets for each device
        device_counts = df["Name"].value_counts()
        print device_counts
            
        return df
    
    def parse_packet(self, pkt, tgt_files_by_src):
        """
        Parses a given packet and extracts the following features:
            - destination MAC address
            - source MAC address
            - time of transmission
            - packet length
            
        The features of the packet are written out to a csv row, which is
        in turn written out to a csv file in the given dictionaries.
        
        This code is heavily based on code written by Capt Steven Beyer.
        
        Parameters
        ----------
        pkt: (Pyshark packet object) the packet from which features will be extracted
        tgt_files_by_src: (dictionary) a dictionary of open csv files.
            The keys are device source addresses, and the values are the open csv files.
        tgt_files_by_dst: (dictionary) a dictionary of open csv files.
            The keys are device destination addresses, and the values are the open csv files.
        """
        try:
            pkt_src = pkt.wlan.sa
            
            if (pkt_src in self.WIFI_DEVICES):
                # Extract features
                pkt_time = pkt.frame_info.time_epoch
                pkt_len = pkt.length
                pkt_subtype_num = pkt.wlan.fc_type_subtype
                
                # Output matches FEATURES
                output = [pkt_time, pkt_len, pkt_src, pkt_subtype_num]
                
                csv.writer(tgt_files_by_src[pkt_src]).writerow(output)            
        
        except AttributeError:
            print "ignored: ", pkt.number
            
    def rename_csv_files(self, device_name):
        directory = self.SRC_DIR
        for filename in os.listdir(directory):
            filename_noextension = os.path.splitext(filename)[0]
            new_filename = device_name[filename_noextension.replace('.',':')] + '.csv'
            os.rename(directory + filename, directory + new_filename)
            
#------------------------------------------------------------------------------
            
if __name__ == "main":
    #Main        
    time_start = time.time()        
    w = WifiPipeline()
    #b.extract_packet_features(filename='/root/Documents/Thesis/PCAPS/wifi-01.cap', create_master=False)
    w.extract_packet_features(create_master=True)
    w.make_dataframe()
    
    # Run One vs All  and One vs One classification strategies
    wifi_features_list = [
            # Packet info
            "PacketLength", "Duration", 
            
            # Vendor 
    #          "Belkin", "Dropcam", "Lifi", "Netgear", "Tp-link",
        
            # 802.11 Data subtype
            "Data", "QoS_Data", "QoS_Null",
    
            # Associated Packets
            "Assoc_Packets"]
    
    wifi_y_list = ["camera", "bulb", "plug"]
    print "One vs all"
    w.one_vs_all_classify(w.df, wifi_features_list, wifi_y_list)
    
    print "One vs one"
    w.one_vs_one_classify(w.df, wifi_features_list, wifi_y_list)
    
    print "Total time:", time.time() - time_start
                
    time_start = time.time()
    b = BLEPipeline()
    #b.extract_packet_features(filename='/root/Documents/Thesis/PCAPS/wifi-01.cap', create_master=False)
    b.extract_packet_features(create_master=True)
    b.make_dataframe()
    
    # Run One vs All  and One vs One classification strategies
    # Run One vs All  and One vs One classification strategies
    ble_features_list = [
    #     'AccessAddr', 'AdvertAddr', 'ScanAddr',
        'BLE_LL_Length', 'TxAddr', 'CompanyID',
    #     'RFChannel',
        'PacketLength', 'Time', 'Assoc_Packets',
        'ADV_DIRECT_IND', 'ADV_IND', 'ADV_NONCONN_IND', 
        'ADV_SCAN_IND', 'CONNECT_REQ', 'SCAN_REQ', 'SCAN_RSP']
    
    ble_y_list = ["door", "lock", "plug", "temp"]
    
    
    print "One vs all"
    b.one_vs_all_classify(b.df, ble_features_list, ble_y_list)
    
    print "One vs one"
    b.one_vs_one_classify(b.df, ble_features_list, ble_y_list)
    
    
    print "Total time:", time.time() - time_start            
                
                