import numpy as np
import librosa
from sklearn import cluster

def feature_extration(x, sr):
    # Zero Crossing Rate
    n0 = 9000
    n1 = 9100
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

    # Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(x, sr)

    # Chroma Frequencies
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

    return np.array(sum(zero_crossings), spectral_centroids, spectral_rolloff, mfccs, chromagram) 
    
if __name__ == "__main__":
    # Load the signal
    x, sr = librosa.load('data/test.wav')
    fitness = feature_extration(x, sr)

    # Clustring 
    kmeans_fit = cluster.KMeans(n_clusters = 3).fit(fitness)
