# dtmf.py: Encodes/decodes dual-tone multi-frequency audio files
import wave, struct, os, copy, platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

class DTMF:
    # encode a DTMF character string into a DTMF audio wavefile
    def encode(aud_file_path=None, dtmf_keys="", key_duration=1, off_duration=1, amp=1, samp_width=2):
        if dtmf_keys is None or dtmf_keys=="": return None
        aud_file_path = DTMF._validate_path_and_type(aud_file_path, ".wav") # validate and get abs file path
        if not key_duration or key_duration < 0.025: raise Exception("Error, minimum key duration is 25 mS!")
        if amp < 0.1 or amp > 1: raise Exception(f"Error, amplitude value must be between 0.1 and 1! Value:{amp}")
        if samp_width != 1 and samp_width != 2: raise Exception(f"Error, sample width must be 1 or 2 bytes! Value:{samp_width}")
        if off_duration and off_duration > 0:
            aud_off_data = DTMF._create_wave_seg(duration=off_duration, sample_rate=44100, freq_list=[], amp=0, arc=False)
        else: aud_off_data = None

        aud_data = []
        for i, ch in enumerate(dtmf_keys):
            if ch == ' ': continue
            if i != 0 and aud_off_data:
                aud_data.extend(aud_off_data)
            if ch.isalpha(): ch = ch.upper()
            if ch not in DTMF.dtmf_tones: raise Exception(f"Error, invalid DTMF character! Char:{ch}")
            freq_list = DTMF.dtmf_tones[ch]
            aud_dtmf_tone = DTMF._create_wave_seg(duration=key_duration, sample_rate=44100, freq_list=freq_list, amp=amp, arc=True)
            aud_data.extend(aud_dtmf_tone)
        DTMF._write_aud_to_file(aud_data, aud_file_path, sample_rate=44100, num_channels=1, samp_width=samp_width)

    # Decodes an DTMF audio wave file (aud_file_path) using a spectrogram (saved as spec_file_path)
    # Returns decoded message and the audio waveform details
    def decode(aud_file_path=None, spec_file_path=None):
        percent_peak_avg_signal_threshold_for_signal_detection = 0.75
        aud_file_path = DTMF._validate_path_and_type(aud_file_path, ".wav") # validate and get abs file path

        aud_wav = DTMF._get_aud_from_wav_file(aud_file_path)
        f, t, Sxx = spectrogram(np.asarray(aud_wav['ch1']), fs=aud_wav['params'].framerate, return_onesided=True, scaling='spectrum', window=('hamming'), noverlap=128, nperseg=512, nfft=1024)

        # slice the spectrogram to only contain frequency between 0 and upper_freq
        upper_i, upper_freq = 0, 2000 # index and desired upper frequency limit for spectrogram
        for i in range(len(f)): # determine the index for freq right above freq limit
            if f[i] > upper_freq:
                upper_i = i
                upper_freq = f[i]
                break
        n_f = f[0:upper_i+1]
        n_Sxx = Sxx[0:upper_i+1]

        # write the spectrogram to file if filename was given
        if spec_file_path:
            spec_file_path = DTMF._validate_path_and_type(spec_file_path, ".png") # validate and get abs file path
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.pcolormesh(t, n_f, n_Sxx)
            plt.savefig(spec_file_path)
            plt.clf()

        # find the closest freq bins to use for dtmf decoding
        freq_bin_i = { 697:0, 770:0, 852:0, 941:0, 1209:0, 1336:0, 1477:0, 1633:0 } # { frequency : freq bin index }
        for freq in freq_bin_i.keys():
            last_diff = 10000
            freq_i = 0
            for i in range(len(n_f)):
                diff = abs(freq-n_f[i])
                if diff < last_diff:
                    last_diff = diff
                    freq_i = i
                else:
                    break
            freq_bin_i[freq] = freq_i

        # get a list of freq peaks (3 total) for each freq_bin and then average them
        freq_bin_peaks = [] # inner list has form: [freq index, peak freq 1, peak freq 2, peak freq 3]
        for freq_i in freq_bin_i.values(): freq_bin_peaks.append([freq_i, 0, 0, 0])
        for t_i in range(len(t)):
            for f_peak in freq_bin_peaks:
                if n_Sxx[ f_peak[0] ][t_i] > f_peak[1]: f_peak[1] = n_Sxx[ f_peak[0] ][t_i]
                elif n_Sxx[ f_peak[0] ][t_i] > f_peak[2]: f_peak[2] = n_Sxx[ f_peak[0] ][t_i]
                elif n_Sxx[ f_peak[0] ][t_i] > f_peak[3]: f_peak[3] = n_Sxx[ f_peak[0] ][t_i]
        freq_bin_avg = {}
        for f_peak in freq_bin_peaks:
            f_avg = sum(f_peak[1:]) / 3
            freq_bin_avg[f_peak[0]] = f_avg # { freq index : freq peak avg }

        # run thru time t searching for dtmf tones using freq_bin_i
        dtmf_chars = []
        cur_char = ''
        for t_i in range(len(t)):
            freq_list = [] # is a list of freq spectrum values with inner lists of form [ freq_index, freq, freq gain ]
            for freq, freq_i in freq_bin_i.items():
                freq_list.append([freq_i, freq, n_Sxx[freq_i][t_i]])
            freq_list.sort(key=lambda x: x[2], reverse=True) # sort by freq signal strength
            if freq_list[0][2] > freq_bin_avg[freq_list[0][0]]*percent_peak_avg_signal_threshold_for_signal_detection and \
               freq_list[1][2] > freq_bin_avg[freq_list[1][0]]*percent_peak_avg_signal_threshold_for_signal_detection: # if there are two signals gt avg peak then there is a tone
                # check if a second tone is present
                if freq_list[2][2] > freq_bin_avg[freq_list[2][0]]*percent_peak_avg_signal_threshold_for_signal_detection and \
                   freq_list[2][2] > freq_bin_avg[freq_list[0][0]]*percent_peak_avg_signal_threshold_for_signal_detection : 
                    raise Exception(f"Error, detected simultaneous DTMF characters!") # if third freq is gt avg peak and near same signal strength as 1st freq then throw exception
                dtmf_char = DTMF.dtmf_chars.get((freq_list[0][1], freq_list[1][1]))
                if dtmf_char and dtmf_char != cur_char:
                    cur_char = dtmf_char
                    dtmf_chars.append(dtmf_char)
            else:
                cur_char = ''
            
        return ''.join(dtmf_chars), copy.copy(aud_wav['params'])

    # throws exception if file_path is not valid; returns absolute path
    def _validate_path_and_type(file_path=None, allowed_ext=None):
        if file_path is None or file_path=="": raise Exception("Error, no file name was given!")
        aud_dir, aud_file = os.path.split(file_path)
        if not os.path.isabs(aud_dir): aud_dir = os.path.abspath(aud_dir) 
        if aud_dir and not os.path.isdir(aud_dir): raise Exception(f"Error, invalid directory for file: {aud_dir}")
        if aud_dir[0] == "\\" and platform.system() == "Windows": raise Exception(f"Error, invalid absolute path for file: {aud_dir}")
        aud_filename, aud_ext = os.path.splitext(aud_file)
        if allowed_ext and aud_ext not in allowed_ext: raise Exception(f"Error, invalid file extension: {aud_ext}")
        if not aud_filename: raise Exception(f"Error, invalid file name: {aud_filename}")
        return os.path.join(aud_dir, aud_file) # re-join with abs dir path

    def _create_wave_seg(duration=0, sample_rate=44100, freq_list=[], amp=1, arc=False):
        delta_t = 1/sample_rate
        omega_list = []
        for freq in freq_list:
            if freq is None or freq <= 0: continue
            omega_list.append(2*np.pi*freq)
        num_samples = duration//delta_t
        time = np.arange(num_samples)*delta_t
        
        # sum all of the freqs together into wav_sum
        wav_sum = [0]*(len(time))
        wav_list = []
        if amp: 
            for omega in omega_list:
                wav_list.append(np.sin(omega*time) * amp)
            for wav in wav_list:
                for i in range(len(wav)):
                    wav_sum[i] += wav[i]
            # normalize sin values so that peak amplitude is at most 1
            peak_val = 1
            for val in wav_sum:
                if abs(val) > peak_val: peak_val = abs(val)
            if peak_val > 1: 
                scale = 1/peak_val
                for i, val in enumerate(wav_sum):
                    wav_sum[i] = wav_sum[i] * scale

        if not arc or not amp:
            return wav_sum
        # add an enter and exit arc ramp to the audio wave
        arc_freq = 1/(duration*2)
        arc_omega = 2*np.pi*arc_freq
        arc_env = np.sin(arc_omega*time)*2
        for i in range(len(wav_sum)):
            if arc_env[i] > 1: continue
            else: wav_sum[i] = wav_sum[i] * arc_env[i]
        return wav_sum

    def _write_aud_to_file(audio, filename, sample_rate=44100, num_channels=1, samp_width=1):
        for val in audio:
            if val > 1 or val < -1: raise Exception(f"Bad audio value:{val}. Must be -1 <= value <= 1")
        # create wave sample data based on samp_width
        if samp_width == 2: 
            int_size = 32767
            samp_data = [(int)(val*int_size) for val in audio]
        elif samp_width == 1: 
            int_size = 127
            samp_data = [(int)((val+1)*int_size) for val in audio] # add offset val by 1 to make all values positive for writing bytes
        else: raise Exception(f"Invalid sample width size:{samp_width}")
        
        # if using dual channels, make a copy of the wave sample data for both channels
        if num_channels == 2:
            aud_data = [0]*(len(samp_data)*2)
            for i in range(0, len(aud_data), 2):
                aud_data[i] = samp_data[i//2] # left channel audio
                aud_data[i+1] = samp_data[i//2] # right channel audio
        elif num_channels == 1:
            aud_data = samp_data # mono channel
        else: raise Exception(f"Invalid number of channels:{num_channels}")

        # open the wave file and write the audio data to it based on samp_width
        obj = wave.open(filename,'wb')
        obj.setnchannels(num_channels)
        obj.setsampwidth(samp_width)
        obj.setframerate(sample_rate)
        if samp_width == 1: obj.writeframesraw(struct.pack('<{}B'.format(len(aud_data)), *aud_data))
        elif samp_width == 2: obj.writeframesraw(struct.pack('<{}h'.format(len(aud_data)), *aud_data))
        obj.close() 

    def _get_aud_from_wav_file(filename, rtn_raw=False):
        obj = wave.open(filename,'rb')
        params = obj.getparams()
        ch1, ch2 = [], []
        chunk_size = 4096

        for size in range(0, params.nframes, chunk_size):
            bytes_read = obj.readframes(chunk_size)
            if params.nchannels == 1 and params.sampwidth == 1:
                ch1.extend(bytes_read)
            elif params.nchannels == 2 and params.sampwidth == 1:
                ch1.extend([bytes_read[i] for i in range(0, len(bytes_read), 2)]) # left channel
                ch2.extend([bytes_read[i] for i in range(1, len(bytes_read), 2)]) # right channel
            elif params.nchannels == 1 and params.sampwidth == 2:
                ch1.extend(struct.unpack('<{}h'.format(len(bytes_read)//2), bytes_read))
            elif params.nchannels == 2 and params.sampwidth == 2:
                chan_bytes = bytearray(b'\x00') * (len(bytes_read)//2)
                for i in range(0, len(bytes_read)//2, 2):
                    chan_bytes[i] = bytes_read[i*2]
                    chan_bytes[i+1] = bytes_read[(i*2)+1]
                ch1.extend(struct.unpack('<{}h'.format(len(bytes_read)//4), chan_bytes)) # left channel
                for i in range(2, len(bytes_read)//2, 2):
                    chan_bytes[i] = bytes_read[i*2]
                    chan_bytes[i+1] = bytes_read[(i*2)+1]
                ch2.extend(struct.unpack('<{}h'.format(len(bytes_read)//4), chan_bytes)) # right channel
            else: raise Exception("Invalid channel and sample width size!")
        obj.close()

        rtn_dict = { 'params':params, 'ch1':ch1, 'ch2':ch2, 'rtn_raw':rtn_raw }
        if rtn_raw:
            return rtn_dict
        
        int_size = (2**(8*params.sampwidth)//2)-1
        for i in range(len(ch1)): ch1[i] = ch1[i]/int_size
        for i in range(len(ch2)): ch2[i] = ch2[i]/int_size 
        return rtn_dict

    dtmf_tones = {
        '1': (697, 1209),
        '2': (697, 1336),
        '3': (697, 1477),
        '4': (770, 1209),
        '5': (770, 1336),
        '6': (770, 1477),
        '7': (852, 1209),
        '8': (852, 1336),
        '9': (852, 1477),
        '0': (941, 1336),
        '*': (941, 1209),
        '#': (941, 1477),
        'A': (697, 1633),
        'B': (770, 1633),
        'C': (852, 1633),
        'D': (941, 1633)
    }
    dtmf_chars = {
        (697, 1209): "1",
        (1209, 697): "1",
        (697, 1336): "2",
        (1336, 697): "2",
        (697, 1477): "3",
        (1477, 697): "3",
        (770, 1209): "4",
        (1209, 770): "4",
        (770, 1336): "5",
        (1336, 770): "5",
        (770, 1477): "6",
        (1477, 770): "6",
        (852, 1209): "7",
        (1209, 852): "7",
        (852, 1336): "8",
        (1336, 852): "8",
        (852, 1477): "9",
        (1477, 852): "9",
        (941, 1336): "0",
        (1336, 941): "0",
        (941, 1209): "*",
        (1209, 941): "*",
        (941, 1477): "#",
        (1477, 941): "#",
        (697, 1633): "A",
        (1633, 697): "A",
        (770, 1633): "B",
        (1633, 770): "B",
        (852, 1633): "C",
        (1633, 852): "C",
        (941, 1633): "D",
        (1633, 941): "D"
    }

if __name__ == "__main__":
    test_dtmf_msg = "abcd0123456789*#"
    print("DTMF Test Message:", test_dtmf_msg)
    DTMF.encode(aud_file_path="dtmf_test.wav", dtmf_keys=test_dtmf_msg, key_duration=0.025, off_duration=0, amp=0.5, samp_width=2)
    print("Created DTMF audio file of message!")
    dec_msg = DTMF.decode(aud_file_path="dtmf_test.wav", spec_file_path="dtmf_spec.png")
    print("Decoded DTMF Msg:", dec_msg)