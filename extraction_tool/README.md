# extraction tool

A simple script to extract individual elements from annotated frames

## usage

1) open the annotation zip file into /annotations 
2) in makesnapshots.py, set video_name variable to the name of the folder containg annotations (example given is called 'video1')
3) run script 
The snapshots will be saved in /annotation_snapshots

note: make sure the /annotation_snapshots and /annotations folders are in the same directory as makesnapshots.py before running it

## dependencies

xmltodict - pip3 install xmltodict
cv2
