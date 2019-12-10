import numpy as np
import os 
import librosa
from sklearn import cluster, metrics

def feature_extration(x, sr):
    # Zero Crossing Rate
    n0 = 9000
    n1 = 9100
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False) 
    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #print("sc",spectral_centroids[0] )

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    #print("sr",spectral_rolloff[0] )

    # Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(x, sr)
    mfccs = mfccs.flatten()
    #print("mfccs",mfccs.shape)

    # Chroma Frequencies
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    chromagram = chromagram.flatten()
    #print("c",chromagram.shape )

    feature = [ x for x_set in [[sum(zero_crossings)], spectral_centroids[:5], spectral_rolloff[:5], mfccs[:5], chromagram[:5]] for x in x_set ]
    return np.array(feature)
    
if __name__ == "__main__":
    # Load train
    train_path = 'data/train'
    train_file = os.listdir(train_path)
    train_features = []
    for train in train_file:
        train = os.path.join(train_path, train)
        train_x, train_sr = librosa.load(train)
        train_feature = feature_extration(train_x, train_sr)
        train_features.append(train_feature)
    #train_true_lables = []

    # Load test
    test_path = 'data/test'
    test_file = os.listdir(test_path)
    test_features = []
    for test in test_file:
        test = os.path.join(test_path, test)
        test_x, test_sr = librosa.load(test)
        test_feature = feature_extration(test_x, test_sr)
        test_features.append(train_feature)
    #test_true_lables = []

    # Clustring 
    range_n_clusters = list(range(2,3))

    for n_clusters in range_n_clusters:
        # Fit
        clusterer = cluster.KMeans(n_clusters=n_clusters).fit(train_features)
        #train_predicts = clusterer.predict(train_features)
        train_labels = clusterer.labels_

        """
        # Accuracy
        count = 0
        num_test = len(test_path)
        test_predicts = clusterer.predict(test_features)
        for predict, label in list(zip(test_predicts, test_labels)):
            if predict == label:
                count += 1
        acc = count / num_test
        """
        
        # For Score
        #h_score = metrics.homogeneity_score(train_true_labels, train_labels)
        #v_score =  metrics.v_measure_score(train_true_labels, train_labels)
        #ar_score = metrics.adjusted_rand_score(train_true_labels, train_labels)
        #ami_score =  metrics.adjusted_mutual_info_score(train_true_labels, train_labels)
        c_score = 0.0#metrics.completeness_score(train_true_labels, train_labels)
        s_score = metrics.silhouette_score(train_features, train_labels, metric='euclidean')
        a_score = 0.0#acc

        print("Epoch 1 for {} clusters :  s_score: {:.4f} c_socre: {:.4f} a_score: {:.4f}  ".format(n_clusters, s_score, c_score, a_score))


