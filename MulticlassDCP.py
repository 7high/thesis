import csv, glob, itertools, os, shutil, time
import numpy as np
import pandas as pd
import pyshark
import matplotlib.pyplot as plt
# Machine Learning modules
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.under_sampling import RandomUnderSampler



class MulticlassDCP():
    
    
    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
    
    def plot_all_confusion_matrices(self, confusion_matrices, y_list):
        classifiers = ['knn', 'lda', 'rf']
        plt.subplots(1, len(classifiers), figsize=(14,4)); # number of rows, number of columns, figure size=(width, height)
        
        for i,c in enumerate(classifiers):    
            plt.subplot(1, len(classifiers), i+1)
            self.plot_confusion_matrix(confusion_matrices[c], classes=y_list, title=c.upper() + ' Confusion Matrix' ,normalize=False);
    
    def plot_all_vcs(self, grid_searches, labels=['knn', 'lda', 'rf']):
        fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=True, figsize=(14, 4))
        fig.text(0.07, 0.5, 'Score', va='center', rotation='vertical')
    
        for i,g in enumerate(grid_searches):    
                plt.subplot(1, len(grid_searches), i+1)
    
                self.plot_vc(g, labels[i])
    
    def plot_vc(self, gs_result, title):
        # Extract mean train and test scores
        mean_train_score, mean_test_score = gs_result['grid_result'].cv_results_['mean_train_score'], gs_result['grid_result'].cv_results_['mean_test_score']
        p_grid = gs_result['grid_result'].param_grid
    
        # Plot mean train and test scores
        parameter = p_grid.keys()[0]
        param_vals = list(p_grid[parameter])
        plt.plot(param_vals, mean_train_score);
        plt.plot(param_vals, mean_test_score);
        plt.title(title.upper())
        plt.xlabel(parameter.capitalize());
        plt.xticks(param_vals);
    
        # Plot best scoring
        train_max = mean_train_score.argmax()
        test_max = mean_test_score.argmax()
        plt.plot(param_vals[train_max], mean_train_score[train_max], 'o');
        plt.plot(param_vals[test_max], mean_test_score[test_max], 'x');
    
    def report_featureimportance(self, feature_importance, features_list):
        # Extract feature importance values
        f_i = pd.Series(feature_importance, index=features_list)
        f_i = f_i.sort_values(ascending=False)
        pd.options.display.float_format = '{:,.5f}'.format
        return f_i
    
    def report_metrics(self, metrics, y_list, csv_name='unnamed', to_csv=True):
        metrics_df = pd.DataFrame()
        metrics_list = ['MCC', 'Mean_Recall', 'Recall', 'Mean_Precision', 'Precision']
    
        # Extract metrics and put into dataframe
        for k,v in metrics.iteritems():
            name = str(k).upper()
            mattcorr = pd.Series({metrics_list[0]:v[0]},name=name)
            bacc = pd.Series({metrics_list[1]:v[1]},name=name)
            recall = pd.Series(v[2],index=[metrics_list[2]+'_'+i.capitalize() for i in y_list], name=name)
            
            ave_prec = pd.Series({metrics_list[3]:v[3]},name=name)
            prec = pd.Series(v[4],index=[metrics_list[4]+'_'+i.capitalize() for i in y_list], name=name)           
            
            metric_series = pd.concat([mattcorr, bacc, recall, ave_prec, prec])
            metrics_df = pd.concat([metrics_df,metric_series], axis=1)
        metrics_df = metrics_df.T
    
        # Print to csv
        if to_csv:
            metrics_df.to_csv('Results/' + csv_name + '.csv', mode='w')
        
        return metrics_df
    
    def run_multiclass(self, df_train, df_test, features_list, y_list, use_tuned=True):
        # Initialize classifiers using tuned hyperparameters
        if use_tuned:
            knn = KNeighborsClassifier(n_neighbors=5)
            lda = LinearDiscriminantAnalysis(n_components=1)
            rf = RandomForestClassifier(max_features=7)  
        else:
            knn = KNeighborsClassifier()
            lda = LinearDiscriminantAnalysis()
            rf = RandomForestClassifier()  
    
        # Train classifiers
        knn_model = knn.fit(df_train[features_list], df_train['DeviceType'])
        lda_model = lda.fit(df_train[features_list], df_train['DeviceType'])
        rf_model = rf.fit(df_train[features_list], df_train['DeviceType'])
        
        # Test classifiers
        knn_preds = knn_model.predict(df_test[features_list])
        lda_preds = lda_model.predict(df_test[features_list])
        rf_preds = rf_model.predict(df_test[features_list])
        
        preds = dict(knn=knn_preds, lda=lda_preds, rf=rf_preds)
        
        # Score classifiers
        metrics = {}
        conf_matrices = {}
        for classifier,pred in preds.iteritems():
            mc = matthews_corrcoef(df_test['DeviceType'], pred)
            
            bacc = balanced_accuracy_score(df_test['DeviceType'], pred)
            recall = recall_score(df_test['DeviceType'], pred, average=None)
            
            ave_prec = precision_score(df_test['DeviceType'], pred, average='macro')
            prec = precision_score(df_test['DeviceType'], pred, average=None)
            
            conf_matrices[classifier] = confusion_matrix(df_test['DeviceType'], pred, labels=y_list)
            metrics[classifier] = mc, bacc, recall, ave_prec, prec
        
        # Feature importance
        feature_importance = rf.feature_importances_
        
        return preds, metrics, conf_matrices, feature_importance
    
    def tune_gridsearch(self, classifier, param_grid, df_train, features_list, y_list): 
        start_time = time.time()
        
        X_train = df_train[features_list]
        y_train = df_train['DeviceType']
        
        grid = GridSearchCV(estimator=classifier, scoring='balanced_accuracy', 
                            param_grid=param_grid, cv=10)   
        
        grid_result = grid.fit(X_train, y_train)
        
        end_time = time.time() - start_time
        return dict(grid_result=grid_result, end_time=end_time)

#------------------------------------------------------------------------------------------------------------
class BLEMulticlassDCP(MulticlassDCP):
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
                'PDUTypeNum','RFChannel', 'PacketLength', 'Time']
    
    path_name = os.getcwd()
    DATE = path_name[path_name.rindex('/')+1:]
    PROC_TIME = "ble_processing_time_" + DATE + ".csv"
    
    def __init__(self):
        """
        Initializes BLEPipeline class. Creates knn, lda, and rf classifiers.
        Contains tuned hyperparameter values.
        Creates a list to store feature importances from rf classifier models.
        """
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.lda = LinearDiscriminantAnalysis(n_components=1, solver='lsqr')
        self.randomforest = RandomForestClassifier(max_features=7)
        self.feature_importances = []
        
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
        
        # One-hot encode PDU_type
        df["PDUType"] = df["PDUTypeNum"].map(self.PDU_TYPES)
        pduType_series = pd.get_dummies(df["PDUType"])
        df = pd.concat([df, pduType_series], axis=1)
        
        #One hot encode RF channel
        rfchannel_series = pd.get_dummies(df["RFChannel"], prefix='Channel')
        df = pd.concat([df, rfchannel_series], axis=1)
        
        # Get number of associated packets for each packet
        list_assoc_pkts = []
        for device in self.BLE_DEVICES:
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
            (BLE LL)
            - device name
            - access address
            - advertising address
            - BLE LL packet length (bytes)
            - PDU type
            - Tx address type (public or random)
            
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
                
                # BLE RF
                rfChannel = pkt.btle_rf.get_field_value('channel')
                
                # Bluetooth
                pktLength = pkt.frame_info.get_field_value('len')
                epochTime = pkt.frame_info.get_field_value('time_epoch')       
                
                # Name as used in thesis document
                name = self.DEVICES_PUBLICADDRS[identifier] if identifier_type == 'advAddr' else self.NAMES_DEVICES[identifier]
                            
                # Output matches the order of FEATURES
                output = [name, deviceName, accessAddr, advAddr, 
                          bleLength, pduType,
                          rfChannel,
                          pktLength, epochTime]
                
                # Write features to csv           
                csv.writer(tgt_files_by_src[name]).writerow(output)
               
        
        except AttributeError:
            print "ignored: ", pkt.number            



#------------------------------------------------------------------------------------------------------------
class WifiMulticlassDCP(MulticlassDCP):
    
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

    FEATURES = ["Time", "PacketLength",
                "SourceAddr","SubtypeNum"]
    
    SRC_DIR = './Wifi_Source/'
    DST_DIR = './Wifi_Destination/'
    PCAP_DIR = '/root/Documents/Thesis/PCAPS'
    TIMING_PKT_NUMBER = 25000
    DATA_FRAME_TYPE = '2'

    path_name = os.getcwd()
    DATE = path_name[path_name.rindex('/')+1:]
    PROC_TIME = "wifi_processing_time_" + DATE + ".csv"
    
    def __init__(self):
        """
        Initializes WifiPipeline class. Creates knn, lda, and rf classifiers.
        Contains tuned hyperparameter values.
        Creates a list to store feature importances from rf classifier models.
        """
        self.knn = KNeighborsClassifier(n_neighbors=9)
        self.lda = LinearDiscriminantAnalysis(n_components=1, solver='lsqr')
        self.randomforest = RandomForestClassifier(max_features=4)
        self.feature_importances = []
    
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
    print "Run as imported module"
                
                