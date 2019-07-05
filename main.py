# -*- coding: utf-8 -*-

import Tkinter as tk
import pyaudio
import wave
import glob, os
from pydub import AudioSegment
from pydub.playback import play
from ScrolledText import ScrolledText
from datetime import datetime
import tkFont as tkfont
import fnmatch
import re

import python_speech_features as mfcc
from sklearn import preprocessing
import numpy as np
import scipy.io.wavfile as wav
from sklearn.mixture import GaussianMixture as GMM

import librosa
from sklearn.neighbors import KNeighborsClassifier





gmm = []
names = []
tr = []
trainFTT = []



class main:

    def __init__(self):


        # Start Tkinter and set Title
        self.root = tk.Tk()
        self.collections = []
        self.root.title('Aluizium 2.1')
        

		# Audio Configuration        
        self.CHUNK = 3024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        # Set Frames
        objects = tk.Frame(self.root, padx=120, pady=20)

        # Pack Frame
        objects.pack(fill=tk.BOTH)



        self.title = tk.Label(objects, text="Speaker recognition software Aluizium 2.1")
        self.button_new_speaker = tk.Button(objects, width=30, padx=10, pady=5, text="Register new speaker", command=lambda: self.create_window1())
        self.button_train = tk.Button(objects, width=30, padx=10, pady=5, text="Train", command=lambda: self.train())
        self.button_recognize_speaker = tk.Button(objects, width=30, padx=10, pady=5, text="Recognize speaker", command=lambda: self.create_window2())
        self.button_exit = tk.Button(objects, width=30, padx=10, pady=5, text='Exit', command=lambda: self.root.destroy())
      
        self.title.grid(row=0, column=0, padx=5, pady=(10,30))
        self.button_new_speaker.grid(row=1, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_train.grid(row=2, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_recognize_speaker.grid(row=3, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_exit.grid(row=4, column=0, columnspan=1, padx=0, pady=(0,10))
        

        tk.mainloop()




    def gettext(self, text):
    	print text.get()




    def play(self, file_name):
    	
        if file_name == 'temp':
            audio = AudioSegment.from_wav(file_name + '.wav')
        else:
            audio = AudioSegment.from_wav('trainset/' + file_name + '.wav')

        play(audio)
        



    def start_record(self, file_name):


        self.st = 1
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        start_time = datetime.now()       
        seconds_before = -1

        print('\n')

        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            self.root.update()

            time_delta = datetime.now() - start_time

            if time_delta.seconds != seconds_before:
                print('Recording... ' + str(time_delta.seconds) + 's')

            seconds_before = time_delta.seconds

        stream.close()

        file_name += '.wav'

        if file_name != 'temp.wav':
            file_name = 'trainset/' + file_name

        print ('Saved as ' + '\'' + file_name + '\'')

        wf = wave.open(file_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()


    def stop(self):
        self.st = 0





    #pega os atributos MFCC de um audio para ser usado no GMM
    def get_MFCC(self, sr, audio):
        #especifica o tamanho do frame, overlap e quantidade de atributos
        features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
        feat     = np.asarray(())
        for i in range(features.shape[0]):
            temp = features[i]
            #nan é atributo nao numerico
            if np.isnan(np.min(temp)):
                continue
            else:
                if feat.size == 0:
                    feat = temp
                else:
                    feat = np.vstack((feat, temp))
        features = feat;
        features = preprocessing.scale(features)

        return features



    def get_FFT(self, audio):
        # x , sr = librosa.load(audio)
        x, sr = librosa.load(audio, sr=16000)
        zero_crossings = librosa.zero_crossings(x, pad=False)
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(x, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)
        result = np.array([np.average(zero_crossings), np.average(spectral_centroids), np.average(spectral_rolloff), np.average(contrast),np.average(bandwidth)])
        return result





    def train(self):

        global gmm
        global names
        global tr
        global trainFTT

        gmm = []
        tr = []
        names = []
        trainFFT = []


        qty = len(fnmatch.filter(os.listdir('trainset/'), '*.wav'))


        #cria 
        for i in range(0, qty):
            gmm.append(GMM(n_components = 8, covariance_type='diag',n_init = 3))
            
        arr = []

        print("\n")

        parent_dir = r'trainset/'
        for wav_file in glob.glob(os.path.join(parent_dir, '*.wav')):
            
            print('Trainning with ' + wav_file)
            
            # MFCC
            (rate,sig) = wav.read(wav_file)

            # FTT
            tr.append(self.get_FFT(wav_file))


            wav_file = re.sub('\.wav$', '', wav_file)
            names.append(wav_file)

            trainset = self.get_MFCC(rate, sig)
            arr.append(trainset)


        trainMFCC = np.array(arr)
        trainFTT = np.array(tr)
         
        for i in range(len(gmm)):
            gmm[i].fit(trainMFCC[i]) #coloca informações pra dentro do gmm, utilizado para treinar

        print(names)






    def test(self):

        global gmm
        global names


       
        aux = np.zeros(len(names))

        score = np.zeros(len(names))
        print('\n')

        (rate,sig) = wav.read("temp.wav")
        temp = self.get_MFCC(rate, sig)
        test = np.array([temp])

        for i in range(len(names)):
            scores = np.array(gmm[i].score(test[0]))
            aux[i] = scores.sum()


        print("Speaker recognized (MFCC): " + names[np.argmax(aux)])


        test = self.get_FFT('temp.wav')
        test = np.array([test])

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(trainFTT, names)

        print("Speaker recognized (TFF):"),
        print(neigh.predict(test[0].reshape(1, -1)))








    def recognize_speaker(self, file_name):
        print(file_name)



    def create_window1(self):
        window1 = tk.Toplevel(padx=120, pady=20)

        self.label_speaker_name = tk.Label(window1, text="Enter new speaker's name:")
        v = tk.StringVar(window1)
        self.entry_speaker_name = tk.Entry(window1, textvariable=v)
        
        self.button_record = tk.Button(window1, width=30, padx=10, pady=5, text='Record', command=lambda: self.start_record(file_name = self.entry_speaker_name.get()))
        self.button_stop = tk.Button(window1, width=30, padx=10, pady=5, text='Stop', command=lambda: self.stop())
        self.button_play = tk.Button(window1, width=30, padx=10, pady=5, text='Play', command=lambda: self.play(file_name = self.entry_speaker_name.get()))
        self.label_text_to_read = tk.Label(window1, text="Text to read: ")


        with open('text.txt', 'r') as file:
            text = file.read()
        
        self.scrolled_text = ScrolledText(window1, height=10, width=80)
        self.scrolled_text.config(wrap="word")
        self.scrolled_text.insert(tk.END, text)

      

        
        self.label_speaker_name.grid(row=0, column=0, padx=5)
        self.entry_speaker_name.grid(row=1, column=0, columnspan=2, padx=0, pady=(0,10))
        self.button_record.grid(row=2, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_stop.grid(row=3, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_play.grid(row=4, column=0, columnspan=1, padx=0, pady=(0,10))
        self.label_text_to_read.grid(row=5, column=0, padx=5, sticky='w', pady=(30,5))
        self.scrolled_text.grid(row=6, column=0, columnspan=1, padx=0, pady=(0,10))

        


    def create_window2(self):
        window2 = tk.Toplevel(padx=120, pady=20)

        self.button_record2 = tk.Button(window2, width=30, padx=10, pady=5, text='Record', command=lambda: self.start_record(file_name = 'temp'))
        self.button_stop2 = tk.Button(window2, width=30, padx=10, pady=5, text='Stop', command=lambda: self.stop())
        self.button_play2 = tk.Button(window2, width=30, padx=10, pady=5, text='Play', command=lambda: self.play(file_name = 'temp'))
        self.button_test = tk.Button(window2, width=30, padx=10, pady=5, text='Test', command=lambda: self.test())
        
        self.button_record2.grid(row=0, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_stop2.grid(row=1, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_play2.grid(row=2, column=0, columnspan=1, padx=0, pady=(0,10))
        self.button_test.grid(row=3, column=0, columnspan=1, padx=0, pady=(0,10))
        










if __name__ == "__main__":
    print ('Initializing software...')
    app = main()
