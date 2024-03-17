import cv2

cap = cv2.VideoCapture("cars.mp4")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    if not ret:
        break

    # compute the absolute difference between the current frame and
    diff = cv2.absdiff(frame1, frame2)

    # Convert images to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh.copy(), None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for contour in contours:
        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < 12000:
            continue


        #Draw bounding box
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    image = cv2.resize(frame1, (width, height))
    out.write(frame1)
    frame1 = frame2

    # Read next frame
    ret, frame2 = cap.read()

cap.release()
out.release()