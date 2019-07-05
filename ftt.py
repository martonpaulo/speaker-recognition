import librosa
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KNeighborsClassifier

nomes = ["Vitor","Marton","Carlos","Adeodato", "Ana", "Filipe", "Matheus"]
trainPath = ["audio\VitorTrain.wav", "audio\MartonTrain.wav", "audio\CarlosTrain.wav", "audio\DeusdatoTrain.wav", "audio\AnaTrain.wav", "audio\FilipeTrain.wav", "audio\MatheusTrain.wav"]
testPath = ["audio\VitorTeste.wav", "audio\MartonTeste.wav", "audio\CarlosTeste.wav", "audio\DeusdatoTeste.wav", "audio\AnaTeste.wav", "audio\FilipeTeste.wav", "audio\MatheusTeste.wav","audio\VitorTeste2.wav","audio\MartonTeste2.wav","audio\cARLOSTeste2.wav","audio\MatheusTeste2.wav"]
def pegaAtributos(audio):
    # x , sr = librosa.load(audio)
    x, sr = librosa.load(audio, sr=16000)
    zero_crossings = librosa.zero_crossings(x, pad=False)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(x, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)
    result = np.array([np.average(zero_crossings), np.average(spectral_centroids), np.average(spectral_rolloff), np.average(contrast),np.average(bandwidth)])
    return result

tr0 = pegaAtributos(trainPath[0])
tr1 = pegaAtributos(trainPath[1])
tr2 = pegaAtributos(trainPath[2])
tr3 = pegaAtributos(trainPath[3])
tr4 = pegaAtributos(trainPath[4])
tr5 = pegaAtributos(trainPath[5])
tr6 = pegaAtributos(trainPath[6])
te0 = pegaAtributos(testPath[0])
te1 = pegaAtributos(testPath[1])
te2 = pegaAtributos(testPath[2])
te3 = pegaAtributos(testPath[3])
te4 = pegaAtributos(testPath[4])
te5 = pegaAtributos(testPath[5])
te6 = pegaAtributos(testPath[6])
te7 = pegaAtributos(testPath[7])
te8 = pegaAtributos(testPath[8])
te9 = pegaAtributos(testPath[9])
te10 = pegaAtributos(testPath[10])

train = np.array([tr0,tr1,tr2,tr3,tr4,tr5,tr6])
test = np.array([te0,te1,te2,te3,te4,te5,te6,te7,te8,te9,te10])

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train, nomes)
print(neigh.predict(test[10].reshape(1, -1)))