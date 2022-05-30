import cv2
import os
import glob
import DATA
from VIDEO import utilsVideo
from HandsTracker import handTracker
from DataFormatting import data_formatting
from tqdm import tqdm
#Change HandsTracker parameters in init and handsfinder
#Change Data number of fingers
#Change Data columns' names in DATAw

def main():
    #Get the path of the current file
    path = os.getcwd()
    #Generate a path for the Videos folder
    path_util = path + '/Videos'
    #Find all the paths for the different video files
    filenames = glob.glob(path_util + "/*.mp4")
    #Get all the files' names
    filenames_util = os.listdir(path_util)
    #Loop through all the files
    for i, filename in enumerate(filenames):
        if filenames_util[i].endswith('.mp4'):
            #Capturing Videos
            cap = cv2.VideoCapture(filename)
            #Executing HandTracking Script
            tracker = handTracker()
            #Getting features
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #Generating path for the transformed videos
            path_video_out = path + '/Tf_Videos/' + filenames_util[i].replace(('.mp4', ''))
            #Write Transformed videos in the right folder
            writer = cv2.VideoWriter(path_video_out, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

            outrR = []
            outlL = []
            out_LR = []
            currenttime = []
            currentframe = 0
            uv = utilsVideo(cap)
            # count the number of frames
            # Get Video Features
            (fps, frame_count, durationSec) = uv.getStats()
            print("\n Total time: {} sec FrameRate: {} FrameCount: {}".format(durationSec, fps, frame_count))
            print("Video: {}, {}/{}".format(filenames_util[i], i+1, len(filenames_util)))

            for j in tqdm(range(frame_count)):
                #Get the current time
                currentframe += 1
                currenttime.append(currentframe/fps)
                #Getting the video and the success
                success, image = cap.read()
                if success:
                    #Getting the current frame and the hands' data
                    image,  outR, outL = tracker.handsFinder(image)
                    #Drawing landmarks on image
                    #out[x or y] [No finger]
                    image = cv2.circle(image, (int(outL[0][0]), int(outL[1][0])), radius=15, color=(0, 0, 255), thickness=-1)
                    image = cv2.circle(image, (int(outL[0][1]), int(outL[1][1])), radius=12, color=(0, 255, 255), thickness=-1)
                    #Saving everything into a csv file
                    DATA.main(outR, outL, outrR, outlL, out_LR, filenames_util[i], currenttime)
                    #Display progress bar
                    image = uv.displayProgressBar(image)
                    #Write a new video
                    #writer.write(image)
                    #Show the video
                    #cv2.imshow("frame", image)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    break

            # MODIFY OL OR OLR IF NEEDED HERE WHAT FILES YOU WANT TO USE
            filename_toformat = '/OutputFileLR/' + filenames_util[i] + '_oLR.txt'
            filename_actions = '/Data/' + filenames_util[i].replace('.mp4', '') + '_distance.txt'
            # Normalize the data
            data_formatting(filename_actions, filename_toformat)
            cap.release()
            writer.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()