import cv2 as cv
import argparse
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import PhotoImage, filedialog, messagebox, simpledialog
from PIL import Image
from PIL import ImageTk

parser = argparse.ArgumentParser(description='Action Recognition Labelling')
parser.add_argument('--use_cpu',default=False, help='Whether to use CPU')
args = parser.parse_args()

if args.use_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Import these after to ensure CPU flag is set properly
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

def extractFrames(vid_paths):

    for vid_path in vid_paths:
        #Extract base video name, and folder path to output frames
        vid_name = os.path.basename(vid_path)
        out_path = os.path.dirname(vid_path)

        #Create videocapture object and extract number of frames to inform user.
        vid = cv.VideoCapture(vid_path)
        num_frames = vid.get(cv.CAP_PROP_FRAME_COUNT)

        #Prompt user for a frame skip value, also informs user that a video is beign processed.
        #Iterate to obtain a valid frame_skip
        frame_skip=-1
        while frame_skip < 0:
            frame_skip = simpledialog.askinteger(title="Video Detected",
                                  prompt="Found video %s in the data directory with a total of %i frames.\n Proceeding to extract frames. Please specify a frame skip value:"%(vid_name, num_frames))
            if frame_skip is None:
                frame_skip=-1

        #Extract every frame_skip'th frame and save to base path
        frame_count = 0
        frame_saved = 0
        ret, frame = vid.read()
        while ret:
            if frame_count % frame_skip == 0:
                #Create name for the extracted frame and build save path
                frame_name = '%s_%i.jpg' % (vid_name, frame_count)
                frame_path = os.path.join(out_path, frame_name)
                cv.imwrite(frame_path, frame)
                frame_saved+=1
            
            frame_count+=1
            ret, frame = vid.read()

        #Inform user that frames have been extracted.
        messagebox.showinfo("Frames Extracted", "Done extracting %i frames from %s" % (frame_saved, vid_name)) 



def process_img():
    #Process image corresponding to img_idx and display
    global img_idx, img_paths, human_idx, show, humans, pose, joints
    
    #Ensure that we haven't run out of images to process
    if img_idx < len(img_paths):

        show = cv.imread(img_paths[img_idx])

        # pose estimation
        humans = estimator.inference(show)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        joints_per_frame = np.array(pose[-1]).astype(np.str)
        
        #Only seperate joints by human if they have been detected
        if len(humans)>0:
            joints= np.array_split(joints_per_frame, len(humans))
        
        human_idx = 0
        #Process Humans
        process_human()
        
    else:
        main_image.configure(image=blankTk)
        main_image.image = blankTk

def process_human():
    global pose, show,humans, human_idx, img_idx
    
    #If for some reason the image doesn't contain humans, ignore
    if len(humans) > 0:
        if human_idx < len(humans):
            xcenters = pose[3]
            bboxes = pose[2]
            height, width = show.shape[:2]
            num_label = "Human: %i" % (human_idx+1)
            xcenter = xcenters[human_idx]
            bbox = bboxes[human_idx]

            out = show.copy()
            cv.rectangle(out, bbox, (0, 250, 20), 2)
            cv.putText(out, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            #resize to display resolution
            tkimg = cv.resize(out, (IMG_WIDTH, IMG_HEIGHT))
            
            #Change BGR to RGB for displaying
            tkimg = cv.cvtColor(tkimg, cv.COLOR_BGR2RGB)

            #Change to PIL format and then to ImageTk
            tkimg = Image.fromarray(tkimg)
            tkimg = ImageTk.PhotoImage(tkimg)
            main_image.configure(image=tkimg)
            main_image.image = tkimg
        else:
            img_idx+=1
            process_img()
    else:
        #No humans were dected, still show the image and then give warning dialog
        out=show.copy()

        #resize to display resolution
        tkimg = cv.resize(out, (IMG_WIDTH, IMG_HEIGHT))
        
        #Change BGR to RGB for displaying
        tkimg = cv.cvtColor(tkimg, cv.COLOR_BGR2RGB)

        #Change to PIL format and then to ImageTk
        tkimg = Image.fromarray(tkimg)
        tkimg = ImageTk.PhotoImage(tkimg)
        main_image.configure(image=tkimg)
        main_image.image = tkimg

        messagebox.showwarning("Warning", "Skipping image since no people are detected.") 
        img_idx+=1
        process_img()

def onLabelClick(label_text):
    global human_idx, joints, img_idx, output_df
    
    #Extract normalized joints per frame
    joints_i = joints[human_idx]
    
    #Construct dataframe entry
    df_entry = np.concatenate([joints_i, np.asarray([label_text])])

    #Catch first label call otherwise index will be nan
    if output_df.values.shape[0] == 0:
        output_df.loc[0] = df_entry
    else:
        output_df.loc[output_df.index.max()+1] = df_entry
    
    #Increase index of image
    human_idx+=1
    process_human()

def add_label():
    lbl = add_lbl_entry.get()
    Labels.append(lbl)
    show_labels(Labels)
    
def show_labels(Labels):
    for i in range(len(Labels)):
        lbl_frame = tk.Frame(master=lbls_frame, borderwidth=1)
        lbl_frame.grid(row=i+1, column=0, padx=5, pady=1)

        lbl = tk.Button(master=lbl_frame, 
                       text='%s' % Labels[i], 
                       bg='grey',
                       width=10,
                       height=1, 
                        command=lambda s=Labels[i]:onLabelClick('%s' % s)) #Needs funky lambda otherwise it won't work
        lbl.pack(padx=5, pady=1)

def browseFiles():
    global data_dir
    data_dir = filedialog.askdirectory(initialdir = "/",
                                          title = "Select a File")
      
    # Change label contents
    top_file_lbl.configure(text=data_dir)

def openDataDir():
    global img_paths, img_idx, IMG_EXTENSIONS, VID_EXTENSIONS
    

    #Get all the names of contents of data dir and create full path to those files
    file_names = os.listdir(data_dir)
    file_paths = [os.path.join(data_dir, file_name) for file_name in file_names]

    #Check if any videos are present in the data directory.
    vids = [file_path for file_path in file_paths if file_path.split('.')[-1] in VID_EXTENSIONS]

    if vids:
        extractFrames(vids)

    #Once any videos have been processed,frames have been writted to data path, need to get list of all images.
    img_idx=0
    img_names = [img_name for img_name in os.listdir(data_dir) if img_name.split('.')[-1] in IMG_EXTENSIONS]
    img_paths = [os.path.join(data_dir, img_name) for img_name in img_names]

    process_img()

def saveDataframe():
    global output_df
    
    output_df.to_csv('./ARL_output.csv')
    
    save_txt.configure(text='Saved to ./ARL_output.csv')



if __name__ == '__main__':

    Labels = []
    data_dir = ''
    img_paths = []
    img_idx = 0
    human_idx = 0

    IMG_EXTENSIONS = ['jpg','png','jpeg']
    VID_EXTENSIONS = ['mp4','mkv','.avi']

    cols = ["nose_x","nose_y","neck_x","neck_y","Rshoulder_x","Rshoulder_y","Relbow_x",
        "Relbow_y","Rwrist_x","RWrist_y","LShoulder_x","LShoulder_y","LElbow_x",
        "LElbow_y","LWrist_x","LWrist_y","RHip_x","RHip_y","RKnee_x","RKnee_y",
        "RAnkle_x","RAnkle_y","LHip_x","LHip_y","LKnee_x","LKnee_y","LAnkle_x",
        "LAnkle_y","REye_x","REye_y","LEye_x","LEye_y","REar_x","REar_y","LEar_x",
        "LEar_y","class"]
    output_df = pd.DataFrame(columns=cols)

    #Image display resolution
    IMG_WIDTH=1080
    IMG_HEIGHT=720

    #Blank Image to show
    blank = Image.fromarray( np.ones([IMG_HEIGHT,IMG_WIDTH,3], dtype=np.uint8)*255)

    #Load estimator
    estimator = load_pretrain_model('VGG_origin')
    #action_classifier = load_action_premodel('Action/framewise_recognition.h5')

    ### GUI    
    root = tk.Tk()
    root.title('Action Recognition Labelling Tool')
    root.iconphoto(False, tk.PhotoImage(file='favicon.png'))

    #### Top Frame ####

    top_frame = tk.Frame(master=root, borderwidth=1)
    top_frame.grid(row=0, column=0)

    top_file_lbl = tk.Label(master=top_frame, 
                       text='Open Data Folder',
                       bg='white', width=50,height=1)
    top_explore_btn = tk.Button(master=top_frame, 
                                text='Browse', 
                                command=browseFiles)

    top_open_btn = tk.Button(master=top_frame, 
                                text='Open', 
                             command=openDataDir)

    top_file_lbl.grid(row=0, column=0)
    top_explore_btn.grid(row=0, column=1)
    top_open_btn.grid(row=0, column=2)

    #### Main Frame ####
    main_frame = tk.Frame(master=root, borderwidth=1)
    main_frame.grid(row=1, column=0)

    ## Label Buttons
    root.columnconfigure(0, weight=1, minsize=75)
    root.rowconfigure(0, weight=1, minsize=75)

    rhs_frame = tk.Frame(master=main_frame, borderwidth=1)
    rhs_frame.grid(row=0, column=1, padx=5, pady=5)

    lbls_frame = tk.Frame(master=rhs_frame, borderwidth=1)
    lbls_frame.grid(row=0,column=0, padx=5,pady=5)

    lbl_name_frame = tk.Frame(master=lbls_frame,relief=tk.RAISED, borderwidth=1)
    lbl_name_frame.grid(row=0, column=0, padx=5, pady=5)
    lbl_name = tk.Label(master=lbl_name_frame,
                        text='Labels',
                       width=10,
                       heigh=1)
    lbl_name.pack()

    show_labels(Labels)

        
    ## Add Label Button
    add_lbl_frame = tk.Frame(master=rhs_frame, borderwidth=1)
    add_lbl_frame.grid(row=1,column=0, padx=5,pady=5)

    add_lbl_name_frame = tk.Frame(master=add_lbl_frame,relief=tk.RAISED, borderwidth=1)
    add_lbl_name_frame.grid(row=0, column=0, padx=5, pady=5)
    add_lbl_name = tk.Label(master=add_lbl_name_frame,
                        text='Add Label',
                       width=10,
                       heigh=1)
    add_lbl_name.pack()

    add_lbl_entry_frame = tk.Frame(master=add_lbl_frame,borderwidth=1)
    add_lbl_entry_frame.grid(row=1, column=0, padx=5, pady=5)
    add_lbl_entry = tk.Entry(master=add_lbl_entry_frame, 
                             fg='black', 
                             bg='grey',
                             width=10)
    add_lbl_entry.pack()

    add_lbl_buttom_frame = tk.Frame(master=add_lbl_frame, borderwidth=1)
    add_lbl_buttom_frame.grid(row=2, column=0, padx=5, pady=5)
    add_lbl_button = tk.Button(master=add_lbl_buttom_frame, 
                               text='Add', 
                               width=5, 
                               bg='grey',
                               command=add_label)
    add_lbl_button.pack()

    # Save Button
    save_btn_frame = tk.Frame(master=rhs_frame, borderwidth=1)
    save_btn_frame.grid(row=2,column=0, padx=5,pady=5)

    save_btn = tk.Button(master=save_btn_frame, 
                         text='Save', 
                         bg='grey', 
                         width=10, 
                         command=saveDataframe)
    save_btn.pack()

    save_txt_frame = tk.Frame(master=rhs_frame, borderwidth=1)
    save_txt_frame.grid(row=3,column=0, padx=5,pady=5)
    save_txt = tk.Label(master=save_txt_frame,
                        fg='red')
    save_txt.pack()

    ### Image frame
    image_frame = tk.Frame(master=main_frame, borderwidth=1)
    image_frame.grid(row=0,column=0, padx=5,pady=1)

    blankTk = ImageTk.PhotoImage(blank)
    main_image = tk.Label(master=image_frame,image=blankTk)
    main_image.image = blankTk
    main_image.pack()
    #user_in = tk.Label(text=entry.get(), width=20, height=5)
    #user_in.pack()
    root.mainloop()