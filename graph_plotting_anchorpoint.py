#!/usr/bin/env python
# coding: utf-8

import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math
from parselmouth.praat import call
from csv import DictReader
import os
import csv
from itertools import zip_longest

def convert_to_log(arr,tonic):
    log_arr=[]
    for i in arr:
        if i==0:
            log_arr.append(0)
        else:
            log_arr.append(np.log2(i/tonic))
    return log_arr

def convert_to_log_for_comparing(arr,tonic):
    tonic_log=np.log2(tonic)
    x=math.modf(tonic_log)
    tonic_log_m=x[0]
    log_arr=[]
    for i in arr:
        if i==0:
            log_arr.append(0)
        else:
            log_arr.append(np.log2(i/tonic_log_m))
    return log_arr

def seperate_mantissa(arr):
    new_arr=[]
    for i in arr:
        x=math.modf(i)
        new_arr.append(x[0])
    return new_arr

def remove_zeros(arr):
    p1=[]
    for i in range(len(arr)):
        if arr[i] == 0:
            p1.append(np.nan)
        else:
            p1.append(arr[i]) 
    return p1

def correct_pitch(pitch_arr,mistakes_arr):
    pitch_arr_n=[]
    for i in range(len(mistakes_arr)):
        if(math.isnan(mistakes_arr[i])==True):
            pitch_arr_n.append(pitch_arr[i])
        else:
            pitch_arr_n.append(np.nan)
    return pitch_arr_n

def compare_log_pitch(t_log_m,s_log_m,teacher_log_pitches_n,student_log_pitches_n):
    half_semitone=1/24*np.log2(2)
    mistakes=[]
    for i in range(len(student_log_pitches_n)):
        diff=abs(t_log_m[i]-s_log_m[i])
        if diff<=half_semitone:
            mistakes.append(np.nan)
        else:
            mistakes.append(student_log_pitches_n[i])
    return mistakes

def extract_pitch_from_path(audio_path,time_step):
    audio = parselmouth.Sound(audio_path)
    pitch_obj = audio.to_pitch(time_step)
    pitch_arr=pitch_obj.selected_array['frequency']
    return pitch_arr  

def extract_pitch_from_array(audio_arr,sr,time_step):
    audio = parselmouth.Sound(audio_arr,sr)
    pitch_obj = audio.to_pitch(time_step)
    pitch_arr=pitch_obj.selected_array['frequency']
    return pitch_arr  

Notes = ['C0', 'C#0', 'D0', 'D#0', 'E0', 'F0', 'F#0', 'G0', 'G#0', 'A0', 'A#0', 'B0','C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1','C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2','C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3','C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4','C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5','C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6','C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7','C8', 'C#8', 'D8', 'D#8', 'E8', 'F8', 'F#8', 'G8', 'G#8', 'A8', 'A#8', 'B8']

def change_tempo(audio_file_path,actual_bpm,desired_bpm):
    factor_bpm=int(actual_bpm)/int(desired_bpm)  
    sound = parselmouth.Sound(audio_file_path)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    duration_tier = call(manipulation, "Extract duration tier")
    call(duration_tier,"Add point", 0,factor_bpm)
    call([duration_tier, manipulation], "Replace duration tier")
    sound_changed_tempo=call(manipulation, "Get resynthesis (overlap-add)")
    return sound_changed_tempo.values,sound_changed_tempo.sampling_frequency

def change_scale_from_array(y,sr,actual_scale,desired_scale):
    factor_scale=get_scale_factor(Notes,actual_scale,desired_scale)  
    sound = parselmouth.Sound(y,sr)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor_scale)
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_changed_scale=call(manipulation, "Get resynthesis (overlap-add)")
    return sound_changed_scale.values, sound_changed_scale.sampling_frequency

def change_scale_from_path(audio,actual_scale,desired_scale):
    factor_scale=get_scale_factor(Notes,actual_scale,desired_scale)  
    sound = parselmouth.Sound(audio)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor_scale)
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_changed_scale=call(manipulation, "Get resynthesis (overlap-add)")
    return sound_changed_scale.values, sound_changed_scale.sampling_frequency

def get_scale_factor(Notes,actual_scale,desired_scale):
    idx_actual=Notes.index(actual_scale)
    idx_desired=Notes.index(desired_scale)
    diff=idx_desired-idx_actual
    factor_scale=2**(diff*(1/12))
    return factor_scale   

def svara_locs_labels(max_pitch,min_pitch,tonic):
    notes = ['S',"",'R',"",'G','M',"",'P',"",'D',"",'N']
    n0 = tonic
    while(n0 >= min_pitch):
        n0=n0/2
    freq_list=[]
    labels_list=[]
    freq_list.append(np.log2(n0/tonic))
    labels_list.append(notes[0])
    i=1
    n=1
    a0=n0
    f0=a0
    while(a0<math.ceil(max_pitch)):
        t0=a0
        if(f0==2*t0):
            a0=2*t0
        f0 = pow(2,n/12)*a0
        freq_list.append(np.log2(f0/tonic))
        labels_list.append(notes[i%12])
        if(n == 12):
            n=1
        else:
            n=n+1
        i=i+1
    return freq_list,labels_list

def extract_teacher_pitch(t_audio,t_scale,t_bpm,s_scale,s_bpm,time_step):
    if(t_scale==s_scale and t_bpm==s_bpm):
         return extract_pitch_from_path(t_audio,time_step)
    elif(t_scale!=s_scale and t_bpm!=s_bpm):
        y,sr=change_tempo(t_audio,t_bpm,s_bpm)
        y1,sr1=change_scale_from_array(y,sr,t_scale,s_scale)
        pitch_arr=extract_pitch_from_array(y1[0],sr1,time_step=0.01)
        return pitch_arr
    elif(t_scale!=s_scale and t_bpm==s_bpm):
        y,sr=change_scale_from_path(t_audio,t_scale,s_scale)
        pitch_arr=extract_pitch_from_array(y[0],sr,time_step=0.01)
        return pitch_arr
    else:
        y,sr=change_tempo(t_audio,t_bpm,s_bpm)
        pitch_arr=extract_pitch_from_array(y[0],sr,time_step=0.01)
        return pitch_arr

def find_pitch_array(t_audio,s_audio,t_scale,t_bpm,s_scale,s_bpm,time_step):
    return extract_teacher_pitch(t_audio,t_scale,t_bpm,s_scale,s_bpm,time_step), extract_pitch_from_path(s_audio,time_step)

def padding(t_arr,s_arr):
    if len(t_arr)<len(s_arr):
        pad = len(s_arr)-len(t_arr)
        t_arr = np.append(t_arr,np.zeros(pad))
        return t_arr,s_arr
    elif len(t_arr)>len(s_arr):
        pad = len(t_arr)-len(s_arr)
        s_arr = np.append(s_arr,np.zeros(pad))
        return t_arr,s_arr
    else:
        return t_arr,s_arr

def get_time_array(pitch):
    pitch_list=pitch
    time_list=[]
    i=float(0.01)
    while(len(time_list)!=len(pitch_list)):
        time_list.append(float(i))
        i=float(i)+float(0.01)
    return np.array(time_list)

def get_max_min_pitch_values(teacher_pitches,student_pitches):
    teacher_max = np.amax(teacher_pitches)
    teacher_min = np.amin(teacher_pitches[teacher_pitches != 0])
    student_max = np.amax(student_pitches)
    student_min = np.amin(student_pitches[student_pitches != 0])
    max_pitch = teacher_max if teacher_max > student_max else student_max
    min_pitch = student_min if student_min < teacher_min else teacher_min
    return max_pitch,min_pitch

def create_time_lst(start,stop,step):
    arr=np.arange(start, stop, step)
    return arr.tolist()        

def x_axis_ticks(bpm,beat_cycle):
    duration_of_beat_cycle = (60/int(bpm))*4
    return  duration_of_beat_cycle

def calucalte_score(mistakes_nan,teacher_log_pitches_nan):
    mistakes=[]
    for i in range(len(teacher_log_pitches_nan)):
        if np.isnan(teacher_log_pitches_nan[i])==False:
            mistakes.append(mistakes_nan[i])
    mistake_count=np.count_nonzero(~np.isnan(mistakes))
    correct_count=len(mistakes)-mistake_count
    
    if (mistake_count + correct_count) == 0:
        return 0
        
    score=(correct_count)/(mistake_count+correct_count)
    return round(score*100)

def calculate_parameters(t_audio,s_audio,t_scale,s_scale,t_bpm,s_bpm,tonic,time_step,t_anchor_point,s_anchor_point,beat_cycle):
    teacher_pitch,student_pitch=find_pitch_array(t_audio,s_audio,t_scale,t_bpm,s_scale,s_bpm,time_step)
    maximum_pitch , minimum_pitch=get_max_min_pitch_values( teacher_pitch,student_pitch)
    teacher_pitch_n,student_pitch_n=padding(teacher_pitch,student_pitch)
    time_arr=get_time_array(teacher_pitch_n)
    duration_of_beat_cycle= x_axis_ticks(s_bpm,beat_cycle)
    
    is_all_zero = np.all((student_pitch == 0))
    if is_all_zero:
        return "No Vocals Found Please Record Again"
    
    teacher_log_pitches,student_log_pitches=convert_to_log(teacher_pitch_n,tonic),convert_to_log(student_pitch_n,tonic)
    teacher_log_pitches_n,student_log_pitches_n=seperate_mantissa(teacher_log_pitches),seperate_mantissa(student_log_pitches)
    t_log,s_log= convert_to_log_for_comparing(teacher_pitch_n,tonic),convert_to_log_for_comparing(student_pitch_n,tonic)
    t_log_m,s_log_m=seperate_mantissa(t_log),seperate_mantissa(s_log)
    
    # ----- SIMULTANEOUS ANCHOR POINT ALIGNMENT -----
    t_anchor_idx = int(t_anchor_point / time_step)
    s_anchor_idx = int(s_anchor_point / time_step)

    t_log_m_aligned = t_log_m[t_anchor_idx:]
    s_log_m_aligned = s_log_m[s_anchor_idx:]
    t_pitches_n_aligned = teacher_log_pitches_n[t_anchor_idx:]
    s_pitches_n_aligned = student_log_pitches_n[s_anchor_idx:]

    min_len = min(len(t_log_m_aligned), len(s_log_m_aligned))
    
    if min_len <= 0:
        return "Audio too short after anchor point applied."

    mistakes = compare_log_pitch(
        t_log_m_aligned[:min_len], 
        s_log_m_aligned[:min_len], 
        t_pitches_n_aligned[:min_len], 
        s_pitches_n_aligned[:min_len]
    )
    
    new_locs , new_labels = svara_locs_labels(maximum_pitch , minimum_pitch ,tonic)
    
    time_limit = time_arr[:min_len][-1] if len(time_arr[:min_len]) > 0 else 0
    x_time_lst = create_time_lst(0, time_limit, duration_of_beat_cycle)

    return (
        time_arr[:min_len], 
        0,
        duration_of_beat_cycle,
        new_locs,
        new_labels, 
        teacher_log_pitches[t_anchor_idx:t_anchor_idx+min_len], 
        student_log_pitches[s_anchor_idx:s_anchor_idx+min_len], 
        mistakes, 
        x_time_lst, 
        0
    )
    
def plot_graph(time_arr, anchor_idx,duration_of_beat_cycle,new_locs,new_labels, teacher_log_pitches,student_log_pitches,mistakes,x_time_lst,delay, save_path=None):
    col = ["#FFFFCB", "#FFFFF0", "#C5C9C7" , "#FFFFF0" , "#C5C9C7" , "#FFFFF0" , "#C5C9C7" , "#FFFFCB" , "#FFFFF0" ,"#C5C9C7" , "#FFFFF0" ,"#C5C9C7" ]
    plt.figure(figsize=(20, 10))
    half_semitone=1/24*np.log2(2)
    
    for i in range(len(x_time_lst)):
        plt.axvline(x_time_lst[i],color='k',linestyle='--')

    plt.setp(plt.gca().axes.get_xticklabels(), rotation=90, horizontalalignment='left')
    
    for i in range(len(new_locs)):
        if i%12 == 0:
            plt.axhline(y=new_locs[i],color="black",linestyle = "dashdot")
    plt.yticks(new_locs,new_labels)
    
    teacher_log_pitches_nan = remove_zeros(teacher_log_pitches)
    student_log_pitches_nan = remove_zeros(student_log_pitches)
    mistakes_nan = remove_zeros(mistakes)
    student_correct_pitch = correct_pitch(student_log_pitches_nan, mistakes)
    
    score = calucalte_score(mistakes, teacher_log_pitches_nan[:len(mistakes)])
    
    teacher_log_pitches_nan_h=[]
    teacher_log_pitches_nan_l=[]
    for i in teacher_log_pitches_nan:
        teacher_log_pitches_nan_h.append(i+1)
        teacher_log_pitches_nan_l.append(i-1)
    
    plt.fill_between(time_arr, np.array(teacher_log_pitches_nan_h)+half_semitone, np.array(teacher_log_pitches_nan_h)-half_semitone, color='#87CEEB', alpha=0.5)
    plt.fill_between(time_arr, np.array(teacher_log_pitches_nan)+half_semitone, np.array(teacher_log_pitches_nan)-half_semitone, color='#87CEEB', alpha=0.5,label ='Reference')
    plt.fill_between(time_arr, np.array(teacher_log_pitches_nan_l)+half_semitone, np.array(teacher_log_pitches_nan_l)-half_semitone, color='#87CEEB', alpha=0.5)
    
    time_shifted = time_arr[:len(student_correct_pitch)]
    plt.plot(time_shifted , student_correct_pitch   , 'o', label ='Student',markersize = 1, color = 'g')
    plt.plot(time_shifted , mistakes_nan  , 'o',label ='Student Mistakes', markersize = 1, color = 'r')
    
    plt.legend(bbox_to_anchor =(0.3, 1),prop={'size': 14}, ncol = 3)
    plt.title("Score: "+str(score)+'%',loc = 'right',size="20")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def process_batch(master_csv_path, teacher_csv_dir, teacher_base_dir, student_base_dir, output_csv_path, save_plots=False, plots_dir=None):
    import pandas as pd
    import os
    import numpy as np

    # 1. READ MASTER MAPPING
    print(f"Reading master mapping: {master_csv_path}")
    if master_csv_path.lower().endswith('.xlsx'):
        df = pd.read_excel(master_csv_path)
    else:
        try:
            df = pd.read_csv(master_csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(master_csv_path, encoding='latin1')
    
    if save_plots and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        
    all_scores = []
    teacher_anchor_maps = {} # Cache for teacher CSV data
        
    for idx, row in df.iterrows():
        try:
            teacher_id = str(row['teacher_id']).strip()
            student_id = str(row['student_id']).strip()
            lesson_name = str(row['lesson_name']).strip()
            t_audio_filename = str(row['t_audio']).strip()
            s_audio_filename = str(row['s_audio']).strip()
            
            teacher_audio_path = os.path.join(teacher_base_dir, teacher_id, 'audio', lesson_name, t_audio_filename)
            student_audio_path = os.path.join(student_base_dir, student_id, 'audio', lesson_name, s_audio_filename)

            if not os.path.exists(teacher_audio_path):
                print(f"[{idx+1}/{len(df)}] ❌ MISSING TEACHER: {teacher_audio_path}")
                all_scores.append(np.nan)
                continue
                
            if not os.path.exists(student_audio_path):
                print(f"[{idx+1}/{len(df)}] ❌ MISSING STUDENT: {student_audio_path}")
                all_scores.append(np.nan)
                continue

            t_scale = str(row['t_scale']).strip()
            s_scale = str(row['s_scale']).strip()
            t_bpm = int(row['t_bpm'])
            s_bpm = int(row['s_bpm'])
            beat_cycle = int(row['beat_cycle'])
            
            # --- GET BOTH ANCHORS ---
            s_anchor_point = float(row['anchor_point'])
            
            # Dynamically load teacher CSV if not already loaded
            if teacher_id not in teacher_anchor_maps:
                t_csv_path = os.path.join(teacher_csv_dir, f"{teacher_id}.csv")
                anchor_map = {}
                if os.path.exists(t_csv_path):
                    print(f"Loading teacher metadata: {t_csv_path}")
                    if t_csv_path.lower().endswith('.xlsx'):
                        t_df = pd.read_excel(t_csv_path)
                    else:
                        try:
                            t_df = pd.read_csv(t_csv_path)
                        except UnicodeDecodeError:
                            t_df = pd.read_csv(t_csv_path, encoding='latin1')
                    for _, t_row in t_df.iterrows():
                        t_audio = str(t_row.get('t_audio_file', '')).strip() 
                        try:
                            anchor = float(t_row.get('anchor_point', 0.0))
                        except (ValueError, TypeError):
                            anchor = 0.0
                        anchor_map[t_audio] = anchor
                else:
                    print(f"⚠️ Warning: Teacher CSV not found at {t_csv_path}")
                
                teacher_anchor_maps[teacher_id] = anchor_map

            # Fallback to student anchor if missing in teacher map
            t_anchor_point = teacher_anchor_maps[teacher_id].get(t_audio_filename, s_anchor_point) 
            
            tonic = librosa.note_to_hz(s_scale)
            time_step = 0.01

            print(f"[{idx+1}/{len(df)}] Scoring: {student_id} -> {t_audio_filename} ...", end=" ")

            # Pass both anchor points to calculate_parameters
            result = calculate_parameters(
                teacher_audio_path, student_audio_path, t_scale, s_scale,
                t_bpm, s_bpm, tonic, time_step, t_anchor_point, s_anchor_point, beat_cycle
            )

            if isinstance(result, str):
                print(f"Failed ({result})")
                all_scores.append(0)
                continue

            time_arr, anchor_idx, duration_of_beat_cycle, new_locs, new_labels, \
            teacher_log_pitches, student_log_pitches, mistakes, x_time_lst, delay = result
            
            teacher_log_pitches_nan = remove_zeros(teacher_log_pitches)
            score = calucalte_score(mistakes, teacher_log_pitches_nan[:len(mistakes)])
            all_scores.append(score)
            
            print(f"Score: {score}%")

            if save_plots and plots_dir:
                plot_filename = f"{student_id}_{lesson_name}_{t_audio_filename.replace('.wav', '')}_{idx}.png"
                plot_save_path = os.path.join(plots_dir, plot_filename)
                
                plot_graph(
                    time_arr, anchor_idx, duration_of_beat_cycle, new_locs, new_labels,
                    teacher_log_pitches, student_log_pitches, mistakes, x_time_lst, delay,
                    save_path=plot_save_path
                )
                
        except Exception as e:
            print(f"\n❌ Error processing row {idx}: {e}")
            all_scores.append(np.nan)

    df['score'] = all_scores
    df.to_csv(output_csv_path, index=False)
    print(f"\n🎉 Batch processing complete! Results saved to {output_csv_path}")

if __name__ == "__main__":
    import pandas as pd
    
    # --- FIXED PATHS HERE ---
    MASTER_CSV = r"C:\dataset\compiled_dataset\scored_mapping_copy.csv"
    TEACHER_CSV_DIR = r"C:\dataset\Data-2\teacher"  #which has t_002.csv t_001.csv metadata files
    TEACHER_DIR = r"C:\dataset\Data-2\teacher"
    STUDENT_DIR = r"C:\dataset\Data-2\student-2"
    OUTPUT_CSV = r"C:\dataset\compiled_dataset\scored_mapping_full_final2.csv"
    SAVE_PLOTS = True
    PLOTS_DIR = r"C:\dataset\Batch_Plot_final"
    
    process_batch(
        master_csv_path=MASTER_CSV,
        teacher_csv_dir=TEACHER_CSV_DIR, 
        teacher_base_dir=TEACHER_DIR,
        student_base_dir=STUDENT_DIR,
        output_csv_path=OUTPUT_CSV,
        save_plots=SAVE_PLOTS,
        plots_dir=PLOTS_DIR
    )