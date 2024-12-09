# Owner avatar Detecting-Face-Parts-and-Overlaying-Masks

## Requirements

- OpenCV 4.0

## Steps to build

To compile on Linux, mac or Windows using Mingw

```
mkdir build
cd build
cmake ..
make
```

## Executables

The following applications are generated.

```
./DetectAndOverlay.cpp

```

Parameters that accepts executable:

```
1. full face
DetectAndOverlay.exe face "D:\AI\code AI\CHAPTER_07\resources\haarcascade_frontalface_alt.xml" "D:\AI\code AI\CHAPTER_07\resources\mask.jpg"

2. Nose
DetectAndOverlay.exe nose "D:\AI\code AI\CHAPTER_07\resources\haarcascade_frontalface_alt.xml" "D:\AI\code AI\CHAPTER_07\resources\haarcascade_mcs_nose.xml" "D:\AI\code AI\CHAPTER_07\resources\nose.png"

3. Mouth
DetectAndOverlay.exe moustache "D:\AI\code AI\CHAPTER_07\resources\haarcascade_frontalface_alt.xml" "D:\AI\code AI\CHAPTER_07\resources\haarcascade_mcs_mouth.xml" "D:\AI\code AI\CHAPTER_07\resources\mouth.png"

4. Ear
DetectAndOverlay.exe ear "D:\AI\code AI\CHAPTER_07\resources\haarcascade_mcs_leftear.xml" "D:\AI\code AI\CHAPTER_07\resources\haarcascade_mcs_rightear.xml"

5. Eyes
DetectAndOverlay.exe eye "D:\AI\code AI\CHAPTER_07\resources\haarcascade_frontalface_alt.xml" "D:\AI\code AI\CHAPTER_07\resources\haarcascade_eye.xml" "D:\AI\code AI\CHAPTER_07\resources\glasses.jpg"

```
