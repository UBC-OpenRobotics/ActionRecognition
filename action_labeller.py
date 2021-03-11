import cv2 as cv
import argparse
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import PhotoImage, filedialog
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
        main_image.configure(image=blankTk)
        main_image.image = blankTk

def onLabelClick(label_text):
    global human_idx, joints, img_idx, output_df
    
    #Extract normalized joints per frame
    joints_i = joints[human_idx]
    
    #Catch first label call otherwise index will be nan
    if output_df.values.shape[0] == 0:
        output_df.loc[0] = [img_idx, joints_i, label_text]
    else:
        output_df.loc[output_df.index.max()+1] = [img_idx, joints_i, label_text]
    
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
    global img_paths, img_idx
    
    img_names = os.listdir(data_dir)
    img_paths = [os.path.join(data_dir, img_name) for img_name in img_names]
    img_idx=0
    
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

    cols = ['img_idx','joint','label']
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