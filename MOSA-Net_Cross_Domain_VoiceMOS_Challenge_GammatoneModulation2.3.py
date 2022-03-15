"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import argparse
import librosa
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io
import scipy.stats
import tensorflow as tf
import os

# from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, \
    AveragePooling1D, TimeDistributed, Dense, Bidirectional, Dropout, \
    GlobalAveragePooling1D, Reshape, Flatten, CuDNNLSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention
from scipy import signal as scp_sig

# Force matplotlib to not use any Xwindows backend.
from gammatone_iir.filter_design import iir_gammatone_filter

matplotlib.use('Agg')

random.seed(999)

epoch = 100
batch_size = 1


def ListRead(filelist):
    f = open(filelist, 'r')
    Path = []
    for line in f:
        Path = Path + [line[0:-1]]
    return Path


def plus_filters(H_1, H_2):
    g_1, p_1 = H_1
    g_2, p_2 = H_2
    return [(g_1 + g_2).real, -(p_1 * g_2 + p_2 * g_1).real], \
           [1., -(p_1 + p_2).real, (p_1 * p_2).real]


# noinspection PyTupleAssignmentBalance,PyTypeChecker
def modulation_filter(cutoff_freq, samp_freq, N=2):
    assert 1 <= len(cutoff_freq) <= 2
    if len(cutoff_freq) == 1:
        f_higher = cutoff_freq[0]

        Wp = f_higher / (samp_freq / 2)
        Ws = (f_higher + f_higher / 2.) / (samp_freq / 2)
        Rp = 3
        Rs = 30
        _, w_B = scp_sig.buttord(Wp, Ws, Rp, Rs)
        bz, ap = scp_sig.butter(N=N, Wn=w_B, btype='lowpass')
        return bz, ap
    else:
        f_lower, f_higher = cutoff_freq
        bandwidth = f_higher - f_lower
        f0 = (f_lower + f_higher) / 2.

        delta_f = bandwidth / 2

        Wp = [(f0 - delta_f) / (samp_freq / 2),
              (f0 + delta_f) / (samp_freq / 2)]
        Ws = [(f0 - delta_f - delta_f) / (samp_freq / 2),
              (f0 + delta_f + delta_f) / (samp_freq / 2)]
        Rp = 3
        Rs = 30
        _, w_B = scp_sig.buttord(Wp, Ws, Rp, Rs)
        bz, ap = scp_sig.butter(N=N, Wn=w_B, btype='bandpass')
        return bz, ap


K = 64
f_min = 60.
f_max = 6000.
fs = 16000
segment_width = 8192
alpha_0 = f_max / f_min
f_c = math.sqrt(f_max * f_min)
H_c_k, H_s_k = [], []
Ha = []
for alpha in [alpha_0 ** (k / K) for k in range(-K // 2, K // 2 + 1)]:
    aH_c, aH_s = iir_gammatone_filter(
        f_c, fs, scale_factor=alpha)
    aH_c = [plus_filters(aH_c[d][0], aH_c[d][1]) for d in range(4)]
    aH_s = [plus_filters(aH_s[d][0], aH_s[d][1]) for d in range(4)]
    H_c_k.append(aH_c)
    H_s_k.append(aH_s)
H_c_k = H_c_k[::-1]
H_s_k = H_s_k[::-1]
bz0, ap0 = modulation_filter(cutoff_freq=[2.], samp_freq=fs, N=2)
bz1, ap1 = modulation_filter(cutoff_freq=[2., 4.], samp_freq=fs, N=2)
bz2, ap2 = modulation_filter(cutoff_freq=[4., 8.], samp_freq=fs, N=2)
bz3, ap3 = modulation_filter(cutoff_freq=[8., 16.], samp_freq=fs, N=2)
bz4, ap4 = modulation_filter(cutoff_freq=[16., 32.], samp_freq=fs, N=2)
bz5, ap5 = modulation_filter(cutoff_freq=[32., 64.], samp_freq=fs, N=3)


def feature_extraction(x_t: np.ndarray):
    x_t = x_t.astype(np.float32)
    total_pad = (
        segment_width - (x_t.shape[-1] % segment_width)) % segment_width
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    x_t = np.pad(x_t, pad_width=((left_pad, right_pad),))

    Xr_kt = np.tile(x_t, reps=(K + 1, 1))
    Xi_kt = np.tile(x_t, reps=(K + 1, 1))
    for k in range(K + 1):
        for d in range(4):
            Xr_kt[k, :] = scp_sig.lfilter(
                b=H_c_k[k][d][0], a=H_c_k[k][d][1], x=Xr_kt[k, :])
            Xi_kt[k, :] = scp_sig.lfilter(
                b=H_s_k[k][d][0], a=H_s_k[k][d][1], x=Xi_kt[k, :])
    A_kt = np.sqrt(np.square(Xr_kt) + np.square(Xi_kt))

    S_mkt = np.empty((6, A_kt.shape[0], A_kt.shape[1]))
    S_mkt[0, :, :] = scp_sig.lfilter(b=bz0, a=ap0, x=A_kt, axis=-1)
    S_mkt[1, :, :] = scp_sig.lfilter(b=bz1, a=ap1, x=A_kt, axis=-1)
    S_mkt[2, :, :] = scp_sig.lfilter(b=bz2, a=ap2, x=A_kt, axis=-1)
    S_mkt[3, :, :] = scp_sig.lfilter(b=bz3, a=ap3, x=A_kt, axis=-1)
    S_mkt[4, :, :] = scp_sig.lfilter(b=bz4, a=ap4, x=A_kt, axis=-1)
    S_mkt[5, :, :] = scp_sig.lfilter(b=bz5, a=ap5, x=A_kt, axis=-1)
    time_len = S_mkt.shape[-1]
    S_mkl = [
        np.mean(
            S_mkt[m, ...].reshape((
                K + 1,
                time_len // (segment_width // (2 ** m)),
                segment_width // (2 ** m)
            )),
            axis=-1)
        for m in range(6)
    ]
    return S_mkl


def Sp_and_phase(path, Noisy=False):
    audio_data, _ = librosa.load(path, sr=16000)


    if np.max(abs(audio_data)) != 0:
        audio_data = audio_data / np.max(abs(audio_data))

    basename = os.path.basename(path).rsplit('.', 1)[0]
    if not os.path.exists(
            f'./data/phase1-main/gammatonemodulation_8192/{basename}.npz'):
        S_mkl = feature_extraction(audio_data)
        S_mblk = [
            np.reshape(S_mkl[0].T, (1, S_mkl[0].shape[1], K + 1)),
            np.reshape(S_mkl[1].T, (1, S_mkl[1].shape[1], K + 1)),
            np.reshape(S_mkl[2].T, (1, S_mkl[2].shape[1], K + 1)),
            np.reshape(S_mkl[3].T, (1, S_mkl[3].shape[1], K + 1)),
            np.reshape(S_mkl[4].T, (1, S_mkl[4].shape[1], K + 1)),
            np.reshape(S_mkl[5].T, (1, S_mkl[5].shape[1], K + 1)),
        ]
        np.savez(
            f'./data/phase1-main/gammatonemodulation_8192/{basename}.npz',
            m0=S_mblk[0], m1=S_mblk[1], m2=S_mblk[2],
            m3=S_mblk[3], m4=S_mblk[4], m5=S_mblk[5])
    else:
        data = np.load(f'./data/phase1-main/gammatonemodulation_8192/{basename}.npz')
        S_mblk = [data[f'm{m}'] for m in range(6)]

    return S_mblk


def train_data_generator(file_list, file_list_hubert, track_name):
    index = 0
    data_path = './data/' + track_name + '/DATA/wav/'
    while True:
        mos_filepath = file_list[index].split(',')
        hubert_filepath = file_list_hubert[index].split(',')

        complete_path = data_path + mos_filepath[0]

        S_mblk = Sp_and_phase(complete_path)
        noisy_hubert = np.load(hubert_filepath[1])

        mos = norm_data(np.asarray(float(mos_filepath[1])).reshape([1]))

        final_len = S_mblk[0].shape[1] + noisy_hubert.shape[1]

        index += 1
        if index == len(file_list):
            index = 0

            random.Random(7).shuffle(file_list)
            random.Random(7).shuffle(file_list_hubert)

        yield [S_mblk[0], S_mblk[1], S_mblk[2], S_mblk[3], S_mblk[4], S_mblk[5],
               noisy_hubert], \
              [mos, mos[0] * np.ones([1, final_len, 1])]


def val_data_generator(file_list, file_list_hubert, track_name):
    index = 0
    data_path = './data/' + track_name + '/DATA/wav/'
    while True:
        mos_filepath = file_list[index].split(',')
        hubert_filepath = file_list_hubert[index].split(',')

        complete_path = data_path + mos_filepath[0]

        S_mblk = Sp_and_phase(complete_path)
        noisy_hubert = np.load(hubert_filepath[1])

        mos = norm_data(np.asarray(float(mos_filepath[1])).reshape([1]))

        final_len = S_mblk[0].shape[1] + noisy_hubert.shape[1]

        index += 1
        if index == len(file_list):
            index = 0

            random.Random(7).shuffle(file_list)
            random.Random(7).shuffle(file_list_hubert)

        yield [S_mblk[0], S_mblk[1], S_mblk[2], S_mblk[3], S_mblk[4], S_mblk[5],
               noisy_hubert], \
              [mos, mos[0] * np.ones([1, final_len, 1])]


def norm_data(input_x):
    input_x = (input_x - 0) / (5 - 0)
    return input_x


def denorm(input_x):
    input_x = input_x * (5 - 0) + 0
    return input_x


def BLSTM_CNN_with_ATT_cross_domain():
    _input_0 = Input(shape=(None, 65))  # T x 65
    _input_1 = Input(shape=(None, 65))  # 2T x 65
    _input_2 = Input(shape=(None, 65))  # 4T x 65
    _input_3 = Input(shape=(None, 65))  # 8T x 65
    _input_4 = Input(shape=(None, 65))  # 16T x 65
    _input_5 = Input(shape=(None, 65))  # 32T x 65

    re_input_5 = Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_5)
    conv1 = (Conv2D(
        2, (2, 2), strides=(1, 1), activation='relu', padding='same')
    )(re_input_5)  # 16T x 65 x 2
    conv1 = (Conv2D(
        2, (2, 2), strides=(2, 1), activation='relu', padding='same')
    )(conv1)  # 16T x 65 x 2
    re_input_4 = concatenate(
        [Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_4), conv1],
        axis=3)  # 16T x 65 x 3

    conv2 = (Conv2D(
        4, (2, 2), strides=(1, 1), activation='relu', padding='same')
    )(re_input_4)  # 8T x 65 x 4
    conv2 = (Conv2D(
        4, (2, 2), strides=(2, 1), activation='relu', padding='same')
    )(conv2)  # 8T x 65 x 4
    re_input_3 = concatenate(
        [Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_3), conv2],
        axis=3)  # 8T x 65 x 5

    conv3 = (Conv2D(
        8, (2, 2), strides=(1, 1), activation='relu', padding='same')
    )(re_input_3)  # 4T x 65 x 8
    conv3 = (Conv2D(
        8, (2, 3), strides=(2, 1), activation='relu', padding='same')
    )(conv3)  # 4T x 65 x 8
    re_input_2 = concatenate(
        [Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_2), conv3],
        axis=3)  # 4T x 65 x 9

    conv4 = (Conv2D(
        16, (2, 2), strides=(1, 1), activation='relu', padding='same')
    )(re_input_2)  # 2T x 65 x 16
    conv4 = (Conv2D(
        16, (2, 3), strides=(2, 1), activation='relu', padding='same')
    )(conv4)  # 2T x 65 x 16
    re_input_1 = concatenate(
        [Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_1), conv4],
        axis=3)  # 2T x 65 x 17

    conv5 = (Conv2D(
        32, (2, 2), strides=(1, 1), activation='relu', padding='same')
    )(re_input_1)  # T x 65 x 32
    conv5 = (Conv2D(
        32, (2, 3), strides=(2, 1), activation='relu', padding='same')
    )(conv5)  # T x 65 x 32
    re_input_0 = concatenate(
        [Reshape((-1, 65, 1), input_shape=(-1, 65))(_input_0), conv5],
        axis=3)  # T x 65 x 33

    conv6 = (Conv2D(
        64, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(re_input_0)  # T x 64 x 64
    conv6 = (Conv2D(
        64, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(conv6)  # T x 64 x 64
    conv6 = (Conv2D(
        64, (3, 3), strides=(1, 3), activation='relu', padding='same')
    )(conv6)  # T x 22 x 64

    conv7 = (Conv2D(
        64, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(conv6)  # T x 22 x 64
    conv7 = (Conv2D(
        64, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(conv7)  # T x 22 x 64
    conv7 = (Conv2D(
        64, (3, 3), strides=(1, 3), activation='relu', padding='same')
    )(conv7)  # T x 8 x 64

    conv8 = (Conv2D(
        128, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(conv7)  # T x 8 x 128
    conv8 = (Conv2D(
        128, (3, 3), strides=(1, 1), activation='relu', padding='same')
    )(conv8)  # T x 8 x 128
    conv8 = (Conv2D(
        128, (3, 2), strides=(1, 2), activation='relu', padding='same')
    )(conv8)  # T x 4 x 128

    re_shape = Reshape((-1, 4 * 128), input_shape=(-1, 4, 128))(conv8)
    _input_hubert = Input(shape=(None, 1024))
    mean_polling = AveragePooling1D(pool_size=2, strides=1, padding='same')(
        _input_hubert)
    bottleneck = TimeDistributed(Dense(512))(mean_polling)
    concat_with_wave2vec = concatenate([re_shape, bottleneck], axis=1)
    blstm = Bidirectional(
        CuDNNLSTM(128, return_sequences=True),
        merge_mode='concat')(concat_with_wave2vec)

    flatten = TimeDistributed(Flatten())(blstm)
    dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1 = Dropout(0.3)(dense1)

    attention = SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        kernel_regularizer=regularizers.l2(1e-4),
        bias_regularizer=regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4, name='Attention')(dense1)
    Frame_score = TimeDistributed(Dense(1, activation='sigmoid'),
                                  name='Frame_score')(attention)
    MOS_score = GlobalAveragePooling1D(name='MOS_score')(Frame_score)

    model = Model(outputs=[MOS_score, Frame_score],
                  inputs=[_input_0, _input_1, _input_2,
                          _input_3, _input_4, _input_5, _input_hubert])

    return model


def train(Train_list, Train_list_hubert, Num_train, Test_list, Test_list_hubert,
          Num_testdata, pathmodel, track_name):
    print('model building...')

    model = BLSTM_CNN_with_ATT_cross_domain()

    get_model_name = pathmodel.split('/')
    model_name = get_model_name[2]

    adam = Adam(lr=1e-6)

    model.compile(loss={'MOS_score': 'mse', 'Frame_score': 'mse'},
                  optimizer=adam)
    plot_model(model, to_file='model_' + str(model_name) + '_GammatoneTimeModulation23_epoch_' + str(
        epoch) + '.png', show_shapes=True)

    if track_name == 'phase1-main':
        print('Running Main-track')
    else:
        print('Load Main-track model as initialized model for OOD-track')
        model.load_weights(
            './PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain-main_GammatoneTimeModulation23_epoch_100.h5')

    with open(pathmodel + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.json',
              'w') as f:  # save the model
        f.write(model.to_json())
    checkpointer = ModelCheckpoint(
        filepath=pathmodel + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.hdf5', verbose=1,
        save_best_only=True, mode='min')

    print('training...')
    g1 = train_data_generator(Train_list, Train_list_hubert, track_name)
    g2 = val_data_generator(Test_list, Test_list_hubert, track_name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # new_model = load_model(
        #     './PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain-main_GammatoneTimeModulation23_epoch_100.h5',
        #     custom_objects={
        #         'SeqSelfAttention': SeqSelfAttention
        #     })
        # hist = new_model.fit_generator(
        #     g1, steps_per_epoch=Num_train, epochs=epoch, verbose=1,
        #     validation_data=g2, validation_steps=Num_testdata, max_queue_size=1,
        #     workers=1, callbacks=[checkpointer])
        # new_model.save(pathmodel + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.h5')
        hist = model.fit_generator(
            g1, steps_per_epoch=Num_train, epochs=epoch, verbose=1,
            validation_data=g2, validation_steps=Num_testdata, max_queue_size=1,
            workers=1, callbacks=[checkpointer])
        model.save(pathmodel + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.h5')

    # plotting the learning curve
    TrainERR = hist.history['loss']
    ValidERR = hist.history['val_loss']
    print(('@%f, Minimun error:%f, at iteration: %i' % (
    hist.history['val_loss'][epoch - 1], np.min(np.asarray(ValidERR)),
    np.argmin(np.asarray(ValidERR)) + 1)))
    print('drawing the training process...')
    plt.figure(2)
    plt.plot(range(1, epoch + 1), TrainERR, 'b', label='TrainERR')
    plt.plot(range(1, epoch + 1), ValidERR, 'r', label='ValidERR')
    plt.xlim([1, epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig(
        'Learning_curve_' + str(model_name) + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.png',
        dpi=150)


def Test(Test_List, Test_List_Hubert_feat, pathmodel, track_name, mode=''):
    print('load model...')

    model_test = BLSTM_CNN_with_ATT_cross_domain()
    with tf.Session() as sess:
        model_test.summary()
        sess.run(tf.global_variables_initializer())
        # model_test.load_weights(pathmodel + '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.h5')
        # model_test.load_weights(
        #     './PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain-main_GammatoneTimeModulation23_epoch_100.h5')
        model_test.load_weights(
            './PreTrained_VoiceMOSChallenge/MOSA-Net_Cross_Domain-main_GammatoneTimeModulation23_epoch_100.hdf5')

        get_model_name = pathmodel.split('/')
        model_name = get_model_name[2]

        print('testing...')
        MOS_Predict = np.zeros([len(Test_List), ])
        MOS_True = np.zeros([len(Test_List), ])

        if mode == 'real-test':
            data_path = './data/' + track_name + '/DATAtest/DATA/wav/'
        else:
            data_path = './data/' + track_name + '/DATA/wav/'
        list_predicted_mos_score = []

        systems_list = list(set([item[0:8] for item in Test_List]))
        MOS_Predict_systems = {system: [] for system in systems_list}
        MOS_True_systems = {system: [] for system in systems_list}

        for i in range(len(Test_List)):
            Asessment_filepath = Test_List[i].split(',')
            hubert_filepath = Test_List_Hubert_feat[i].split(',')
            wav_name = Asessment_filepath[0]

            complete_path = data_path + wav_name
            S_mblk = Sp_and_phase(complete_path)
            noisy_hubert = np.load(hubert_filepath[1])

            mos = float(Asessment_filepath[1])

            [MOS_1, frame_mos] = model_test.predict(
                [S_mblk[0], S_mblk[1], S_mblk[2],
                 S_mblk[3], S_mblk[4], S_mblk[5], noisy_hubert], verbose=0,
                batch_size=batch_size)

            denorm_MOS_predict = denorm(MOS_1)
            MOS_Predict[i] = denorm_MOS_predict
            MOS_True[i] = mos

            system_names = wav_name[0:8]

            MOS_Predict_systems[system_names].append(denorm_MOS_predict[0])
            MOS_True_systems[system_names].append(mos)

            estimated_score = denorm_MOS_predict[0]
            info = Asessment_filepath[0] + ',' + str(estimated_score[0])
            list_predicted_mos_score.append(info)

    MOS_Predict_systems = np.array(
        [np.mean(scores) for scores in MOS_Predict_systems.values()])
    MOS_True_systems = np.array(
        [np.mean(scores) for scores in MOS_True_systems.values()])

    with open(f'List_predicted_score_mos{mode}' + str(model_name) + '_GammatoneTimeModulation23_epoch_' + str(
            epoch) + '_answer.txt', 'w') as g:
        for item in list_predicted_mos_score:
            g.write("%s\n" % item)

    print('Utterance Level-Score')
    MSE = np.mean((MOS_True - MOS_Predict) ** 2)
    print('Test error= %f' % MSE)
    LCC = np.corrcoef(MOS_True, MOS_Predict)
    print('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC = scipy.stats.spearmanr(MOS_True.T, MOS_Predict.T)
    print('Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU = scipy.stats.kendalltau(MOS_True, MOS_Predict)
    print(('Kendalls tau correlation= %f' % KTAU[0]))
    print('')

    # Plotting the scatter plot
    M = np.max([np.max(MOS_Predict), 5])
    plt.figure(1)
    plt.scatter(MOS_True, MOS_Predict, s=14)
    plt.xlim([0, M])
    plt.ylim([0, M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('LCC= %f, SRCC= %f, MSE= %f, KTAU= %f' % (
        LCC[0][1], SRCC[0], MSE, KTAU[0]))
    plt.show()
    plt.savefig(
        ('Scatter_plot_MOS_' + str(model_name) +
         '_GammatoneTimeModulation23_epoch_' + str(epoch) + '.png'),
        dpi=150)

    print('Systems Level-Score')
    MSE = np.mean((MOS_True_systems - MOS_Predict_systems) ** 2)
    print('Test error= %f' % MSE)
    LCC = np.corrcoef(MOS_True_systems, MOS_Predict_systems)
    print('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC = scipy.stats.spearmanr(MOS_True_systems.T, MOS_Predict_systems.T)
    print('Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU = scipy.stats.kendalltau(MOS_True_systems, MOS_Predict_systems)
    print('Kendalls tau correlation= %f' % KTAU[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--name', type=str,
                        default='MOSA-Net_Cross_Domain-main')
    parser.add_argument('--track', type=str, default='phase1-main')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    pathmodel = './PreTrained_VoiceMOSChallenge/' + str(args.name)
    track_name = args.track

    Train_list = ListRead(
        './data/' + track_name + '/DATA/sets/train_mos_list.txt')
    Train_list_hubert = ListRead(
        './data/' + track_name + '/DATA/sets/List_Npy_Train_hubert_MOS_Challenge_phase1_main.txt')
    NUM_DATA_TRAIN = len(Train_list)

    random.Random(7).shuffle(Train_list)
    random.Random(7).shuffle(Train_list_hubert)

    Val_list = ListRead('./data/' + track_name + '/DATA/sets/val_mos_list.txt')
    Val_list_hubert = ListRead(
        './data/' + track_name + '/DATA/sets/List_Npy_Val_hubert_MOS_Challenge_phase1_main.txt')
    NUM_DATA_VAL = len(Val_list)

    # Test_List = Val_list
    # Test_List_Hubert_feat = Val_list_hubert
    Test_List = ListRead('./data/' + track_name + '/DATAtest/DATA/sets/test_mos_list.txt')
    Test_List_Hubert_feat = ListRead(
        './data/' + track_name + '/DATAtest/DATA/sets/List_Npy_Test_hubert_MOS_Challenge_phase1_main.txt')

    if args.mode == 'train':
        print('training')
        train(Train_list, Train_list_hubert, NUM_DATA_TRAIN, Val_list,
              Val_list_hubert, NUM_DATA_VAL, pathmodel, track_name)
        print('complete training stage')
    if args.mode == 'test':
        print('testing')
        Test(Test_List, Test_List_Hubert_feat, pathmodel, track_name)
        print('complete testing stage')
    if args.mode == 'real-test':
        print('Testing on test data')
        Test(Test_List, Test_List_Hubert_feat, pathmodel, track_name, mode='real-test')
        print('complete testing stage')