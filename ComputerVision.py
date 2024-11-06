import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
#import scipy.io
import plotly.graph_objects as go

"""ricorda una cosa: di entrambe le scacchiere ci interessa solo l'angolo yaw

CAMERA CALIBRATION: FIRST CAMERA (quella messa sullo schienale di dietro)
convenzione: variabili che si riferiscono alla prima camera non hanno apici
"""

images_paths = glob.glob("calibration con sedia CAMERA1/*.png") #glob: questa funzione va in una cartella e cerca i file dentro. * indica "tutto quello dentro la cartella", non Ã¨ in ordine alfabetico. aggiungendo .png mi prende solo le foto e non i video
#print(images_paths)
print(f'the number of pictures of the calibration of the first camera is {len(images_paths)}')

for k,path in enumerate(images_paths):
  image = cv.imread(path)
  #print(image.shape)
  #plt.imshow(image)
  #plt.axis("off")
  #plt.title(f"image number {k+1}")
  #plt.show()

CHECKERBOARD = (9,6)  #this notation represents a tuple. a tuple is a collection of elements (the elements can be of different type)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

objp = []  #inizialization of a list. This list will store the coordinates of the corners of the checkerboard in 3D space
for i in range(CHECKERBOARD[1]):
  for j in range(CHECKERBOARD[0]):
    objp.append([j*25.0, i*25.0, 0.0]) #the function append() is used to add an element to the end of a list. this line appends the coordinate [j*15.0, i*15.0, 0.0] to objp. We are assuming that the checkerboard is lying flat on the XY plane with Z coordinate being zero. we multiply by 15 because the distance from a corner to another is of 15mm
#print(objp)
objp = np.array(objp, np.float32)

object_points = []
images_points = []

for k, path in enumerate(images_paths):
  image = cv.imread(path)
  #print(image.shape)
  gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  #print(gray.shape)
  #con questa prima parte andiamo a leggere ogni immagine. il primo print stampa le dimensioni dell'immagine a colori, dopo di che la rendiamo in bianco e nero e il secondo print stampa le dimensioni dell'immagine in bianco e nero. i primi due numeri restano uguali, il terzo cambia

  ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
  #print(ret, len(corners))
  #len(corners) print the number of corners found in the image

  #questo if mi dice "se gli angoli sono stati effettivamente trovati nella foto..."
  if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
    images_points.append(corners2)
    object_points.append(objp)

    #disegno i punti trovati sull'immagine
    image = cv.drawChessboardCorners(image, CHECKERBOARD, corners2, True)
    #plt.imshow(image)
    #plt.axis("off")
    #plt.show()

ret, intrinsic, dist, r_vecs, t_vecs = cv.calibrateCamera(object_points, images_points, (1213, 1936), None, None)
#print(ret)  #printa l'errore della calibrazione in pixel#
#print(intrinsic)
#print(dist)
#print(r_vecs)
#print(t_vecs)

"""CAMERA CALIBRATION: SECOND CAMERA (quella messa sullo schienale di lato)
le variabili che si riferiscono alla seconda camera hanno apice 2
"""

images2_paths = glob.glob("calibration con sedia CAMERA2/*.png") #glob: questa funzione va in una cartella e cerca i file dentro. * indica "tutto quello dentro la cartella", non Ã¨ in ordine alfabetico. aggiungendo .png mi prende solo le foto e non i video
#print(images2_paths)
print(f'the number of pictures of the calibration of the second camera is {len(images2_paths)}')

for k,path in enumerate(images2_paths):
  image = cv.imread(path)
  #print(image.shape)
  #plt.imshow(image)
  #plt.axis("off")
  #plt.title(f"image number {k+1}")
  #plt.show()

CHECKERBOARD2 = (4,3)  #this notation represents a tuple. a tuple is a collection of elements (the elements can be of different type)
criteria2 = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

objp2 = []  #inizialization of a list. This list will store the coordinates of the corners of the checkerboard in 3D space
for i in range(CHECKERBOARD2[1]):
  for j in range(CHECKERBOARD2[0]):
    objp2.append([j*24.0, i*24.0, 0.0]) #the function append() is used to add an element to the end of a list. this line appends the coordinate [j*15.0, i*15.0, 0.0] to objp. We are assuming that the checkerboard is lying flat on the XY plane with Z coordinate being zero. we multiply by 15 because the distance from a corner to another is of 15mm
#print(objp2)
objp2 = np.array(objp2, np.float32)

object2_points = []
images2_points = []

for k, path in enumerate(images2_paths):
  image = cv.imread(path)
  #print(image.shape)
  gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  #print(gray.shape)
  #con questa prima parte andiamo a leggere ogni immagine. il primo print stampa le dimensioni dell'immagine a colori, dopo di che la rendiamo in bianco e nero e il secondo print stampa le dimensioni dell'immagine in bianco e nero. i primi due numeri restano uguali, il terzo cambia

  ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD2, None)
  #print(ret, len(corners))
  #len(corners) print the number of corners found in the image

  #questo if mi dice "se gli angoli sono stati effettivamente trovati nella foto..."
  if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria2)
    images2_points.append(corners2)
    object2_points.append(objp2)

    #disegno i punti trovati sull'immagine
    image = cv.drawChessboardCorners(image, CHECKERBOARD2, corners2, True)
    #plt.imshow(image)
    #plt.axis("off")
    #plt.show()

ret2, intrinsic2, dist2, r2_vecs, t2_vecs = cv.calibrateCamera(object2_points, images2_points, (1213, 1936), None, None)
#print(ret2)  #printa l'errore della calibrazione in pixel#
#print(intrinsic2)
#print(dist2)
#print(r2_vecs)
#print(t2_vecs)

"""# VIDEO ANALYSIS 1 (della scacchiera sul retro dello schienale)"""

video_path = "videoAnalysis1/scacchiera90.mp4"
cap = cv.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS) #frame per second
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    time_interval = 1 / fps
    print("camera1: Time interval between consecutive frames:", time_interval, "seconds")
    print(f"camera1: Video FPS: {fps}")
    print(f"camera1: Frame count: {frame_count}")
    print(f"camera1: Video duration (seconds): {duration}")

# define the time vector
timestamps = []
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    timestamps.append(cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0)  # Convert milliseconds to seconds

cap.release()
#print(timestamps)

CHECKERBOARD = (9,6)  #this notation represents a tuple. a tuple is a collection of elements (the elements can be of different type)
#to store the history of the translations along the x, y, and z axes. These lists will store the position of the camera in 3D space over time.

x_history = []
y_history = []
z_history = []
roll_history = []
pitch_history = []
yaw_history = []

cap = cv.VideoCapture(video_path)
while cap.isOpened():  #cap is open and the frames are available
  ret, frame = cap.read()


  if not ret:
    break

  gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
  ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

  if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)

    ret, r_vec, t_vec = cv.solvePnP(objp, corners2, intrinsic, dist) #r_vec and t_vec represent the rotation and translation vectors of the camera with respect to the chessboard pattern detected in the image
    x_history.append(t_vec[0])
    y_history.append(t_vec[1])
    z_history.append(t_vec[2])

    r_mat, _ = cv.Rodrigues(r_vec)  # This function converts the rotation vector (r_vec) into a rotation matrix (r_mat). The rotation vector is a compact representation of a 3D rotation, and cv.Rodrigues is used to convert it to a 3x3 rotation matrix.
    sy = np.sqrt(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2) #radice quadrata delle somme dei valori della prima riga della matrice

    singular = sy < 1e-6  #se accade questa condizione yaw e pitch diventano indistinguibili quindi bisogna suddividere i casi

    if not singular:
        roll = np.arctan2(r_mat[2, 1], r_mat[2, 2])
        pitch = np.arctan2(-r_mat[2, 0], sy)
        yaw = np.arctan2(r_mat[1, 0], r_mat[0, 0])
    else:
        roll = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
        pitch = np.arctan2(-r_mat[2, 0], sy)
        yaw = 0

    roll_history.append(roll)
    pitch_history.append(pitch)
    yaw_history.append(yaw)

cap.release()

time_history = [i / fps for i in range(len(roll_history))]

roll_history_deg = []
pitch_history_deg = []
yaw_history_deg = []
roll_history_deg = np.degrees(roll_history)
pitch_history_deg = np.degrees(pitch_history)
yaw_history_deg = np.degrees(yaw_history)

length_of_yaw = yaw_history_deg.size
print("Length of yaw_history_deg vector:", length_of_yaw)
#+print(yaw_history_deg)


# Plot delle posizioni
plt.plot(time_history, x_history, label='x')
plt.plot(time_history, y_history, label='y')
plt.plot(time_history, z_history, label='z')
plt.legend()
plt.title('Position of the first camera')
plt.show()

#plot degli angoli
plt.plot(time_history, roll_history_deg, label='roll')
plt.plot(time_history, pitch_history_deg, label='pitch')
plt.plot(time_history, yaw_history_deg, label='yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Angles of the first camera')
plt.show()

#plot degli angoli (solo yaw)
plt.plot(time_history, yaw_history_deg, label='yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Yaw of the first camera')
plt.show()

yaw_history_deg_meno = -yaw_history_deg
yaw_history_deg_meno = yaw_history_deg_meno - 90
print(yaw_history_deg_meno)
if(yaw_history_deg_meno > 269).any():
    yaw_history_deg_meno = yaw_history_deg_meno - 269.4


#plot degli angoli (solo yaw)
plt.plot(time_history, yaw_history_deg_meno, label='yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Yaw of the first camera')
plt.show()

"""
#flattens the list x_history, y_history and z_history into a 1D numpy array.
x = np.array(x_history).reshape(-1)
y = np.array(y_history).reshape(-1)
z = np.array(z_history).reshape(-1)

import plotly.graph_objects as go #library for creating interactive visualizations in Python
import plotly

fig = go.Figure(data = [
    go.Scatter3d(x = x, y = y, z = z, mode = "markers")
])

fig.show()
"""

# to save the variables in python
np.savez('variables_def6_camera1.npz',timeInterval=time_interval, yaw =yaw_history_deg,
         roll=roll_history_deg, pitch=pitch_history_deg)

"""
# to save the variables in matlab
scipy.io.savemat('variables_def6_camera1.mat', {'timeInterval': time_interval, 'yaw': yaw_history_deg,
    'roll': roll_history_deg, 'pitch': pitch_history_deg})
"""

"""
savemat('variables_def1_camera1.mat', {
    'timeInterval': time_interval,
    'yaw': yaw_history_deg,
    'roll': roll_history_deg,
    'pitch': pitch_history_deg
})
"""

"""# VIDEO ANALYSIS 2 (della scacchiera sul lato dello schienale)"""

video2_path = "videoAnalysis1/videoAnalysis2/def3_camera2.avi"
cap2 = cv.VideoCapture(video2_path)

# Check if the video was successfully opened
if not cap2.isOpened():
    print(f"Error: Could not open video file {video2_path}")
else:
    # Get video properties
    fps2 = cap2.get(cv.CAP_PROP_FPS) #frame per second
    frame2_count = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))
    duration2 = frame2_count / fps2
    time2_interval = 1 / fps2
    print("camera2: Time interval between consecutive frames:", time2_interval, "seconds")
    print(f"camera2: Video FPS: {fps2}")
    print(f"camera2: Frame count: {frame2_count}")
    print(f"camera2: Video duration2 (seconds): {duration2}")


# define the time vector
timestamps2 = []
frames2 = []

while True:
    ret, frame = cap2.read()
    if not ret:
        break
    frames2.append(frame)
    timestamps2.append(cap2.get(cv.CAP_PROP_POS_MSEC) / 1000.0)  # Convert milliseconds to seconds

cap2.release()
#print(timestamps2)

CHECKERBOARD2 = (4,3)  #this notation represents a tuple. a tuple is a collection of elements (the elements can be of different type)
#to store the history of the translations along the x, y, and z axes. These lists will store the position of the camera in 3D space over time.

x2_history = []
y2_history = []
z2_history = []
roll2_history = []
pitch2_history = []
yaw2_history = []

cap2 = cv.VideoCapture(video2_path)
while cap2.isOpened():  #cap is open and the frames are available
  ret, frame = cap2.read()


  if not ret:
    break

  gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
  ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD2, None)

  if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria2)

    ret, r_vec, t_vec = cv.solvePnP(objp2, corners2, intrinsic2, dist2) #r_vec and t_vec represent the rotation and translation vectors of the camera with respect to the chessboard pattern detected in the image
    x2_history.append(t_vec[0])
    y2_history.append(t_vec[1])
    z2_history.append(t_vec[2])

    r_mat, _ = cv.Rodrigues(r_vec)  # This function converts the rotation vector (r_vec) into a rotation matrix (r_mat). The rotation vector is a compact representation of a 3D rotation, and cv.Rodrigues is used to convert it to a 3x3 rotation matrix.
    sy = np.sqrt(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2) #radice quadrata delle somme dei valori della prima riga della matrice

    singular = sy < 1e-6  #se accade questa condizione yaw e pitch diventano indistinguibili quindi bisogna suddividere i casi

    if not singular:
        roll = np.arctan2(r_mat[2, 1], r_mat[2, 2])
        pitch = np.arctan2(-r_mat[2, 0], sy)
        yaw = np.arctan2(r_mat[1, 0], r_mat[0, 0])
    else:
        roll = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
        pitch = np.arctan2(-r_mat[2, 0], sy)
        yaw = 0

    roll2_history.append(roll)
    pitch2_history.append(pitch)
    yaw2_history.append(yaw)

cap2.release()

time2_history = [i / fps2 for i in range(len(roll2_history))]

roll2_history_deg = []
pitch2_history_deg = []
yaw2_history_deg = []
roll2_history_deg = np.degrees(roll2_history)
pitch2_history_deg = np.degrees(pitch2_history)
yaw2_history_deg = np.degrees(yaw2_history)

length_of_yaw2 = yaw2_history_deg.size
print("Length of yaw_history_deg vector:", length_of_yaw2)
#print(yaw2_history_deg)

# Plot delle posizioni
plt.plot(time2_history, x2_history, label='x')
plt.plot(time2_history, y2_history, label='y')
plt.plot(time2_history, z2_history, label='z')
plt.legend()
plt.title('Positions of the second camera')
plt.show()

#plot degli angoli
plt.plot(time2_history, roll2_history_deg, label='roll')
plt.plot(time2_history, pitch2_history_deg, label='pitch')
plt.plot(time2_history, yaw2_history_deg, label='yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Angles of the second camera')
plt.show()

#plot degli angoli (solo yaw)
plt.plot(time2_history, yaw2_history_deg, label='yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Yaw of the second camera')
plt.show()


"""
#flattens the list x_history, y_history and z_history into a 1D numpy array.
x = np.array(x2_history).reshape(-1)
y = np.array(y2_history).reshape(-1)
z = np.array(z2_history).reshape(-1)

import plotly.graph_objects as go #library for creating interactive visualizations in Python
import plotly

fig = go.Figure(data = [
    go.Scatter3d(x = x, y = y, z = z, mode = "markers")
])

fig.show()
"""

np.savez('variables_def6_camera2.npz',timeInterval=time2_interval, yaw =yaw2_history_deg,
         roll=roll2_history_deg, pitch=pitch2_history_deg)

"""
# to save the variables in matlab
scipy.io.savemat('variables_def6_camera2.mat', {'timeInterval': time2_interval, 'yaw': yaw2_history_deg,
    'roll': roll2_history_deg, 'pitch': pitch2_history_deg})
"""

"""
savemat('variables_def1_camera2.mat', {
    'timeInterval': time2_interval,
    'yaw': yaw2_history_deg,
    'roll': roll2_history_deg,
    'pitch': pitch2_history_deg
})
"""


"""# Confronto tra le due camere
tieni presente che lo Yaw della camera 1 corrisponde al roll della camera 2.

invece Yaw di camera 2 corrispone al roll di camera 1
"""

plt.plot(time_history, roll_history_deg, label='roll 1')
plt.plot(time2_history, yaw2_history_deg, label='yaw 2')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.show()

plt.plot(time_history, yaw_history_deg, label='yaw 1')
plt.plot(time2_history, roll2_history_deg, label='roll 2')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.show()
