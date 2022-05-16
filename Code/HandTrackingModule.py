import cv2
import os
import glob
import DATA
from VIDEO import utilsVideo
from HandsTracker import handTracker


def main():
    path = os.getcwd()
    path_util = path + '/Videos'
    filenames = glob.glob(path + '/Videos' + "/*.mp4")
    filenames_util = os.listdir(path_util)
    for i, filename in enumerate(filenames):
        if filenames_util[i].endswith('.mp4'):
            #Using Video
            cap = cv2.VideoCapture(filename)
            #Executing HandTracking Script
            tracker = handTracker()
            #Saving Video

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            path_video_out= path + '/Tf_Videos' + filenames_util[i]

            print(filename)
            writer = cv2.VideoWriter(path_video_out, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

            outrR = []
            outlL = []
            out_LR = []
            currenttime = []
            currentframe = 0
            uv = utilsVideo(cap)
            # count the number of frames
            # calculate duration of the video
            (fps, frame_count, durationSec) = uv.getStats()
            print("Total time: {} sec FrameRate: {} FrameCount: {}".format(durationSec, fps, frame_count))
            while True:
                #Get the current time
                currentframe += 1
                currenttime.append(currentframe/fps)
                #Getting the video and the success
                success, image = cap.read()
                if success:
                    #Getting the current frame and the hands' data
                    image,  outR, outL = tracker.handsFinder(image)
                    #Saving everything into a csv file
                    DATA.main(outR, outL, outrR, outlL, out_LR, filenames_util[i], currenttime)
                    #Show progress bar
                    image = uv.displayProgressBar(image)
                    #Write a new video
                    writer.write(image)
                    #Show the video
                    cv2.imshow("frame", image)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()