import numpy as np
import cv2

def show_keypoints(x, y_true, y_pred):
    
    im = (x[0].transpose(1,2,0)).copy()
    pts = y_true[0]
    pred_pts = y_pred[0]
    true_x = [pts[0],pts[2],pts[4],pts[6],pts[8]]
    true_y = [pts[1],pts[3],pts[5],pts[7],pts[9]]
    pred_x = [pred_pts[0],pred_pts[2],pred_pts[4],pred_pts[6],pred_pts[8]]
    pred_y = [pred_pts[1],pred_pts[3],pred_pts[5],pred_pts[7],pred_pts[9]]
    cv2.imshow('keypoints', im)
    for xt,yt,xp,yp in zip(true_x, true_y, pred_x, pred_y):
        cv2.circle(im ,(int(xt),int(yt)),10,(255,0,0),-1)
        cv2.circle(im ,(int(xp),int(yp)),10,(0,0,255),-1)
    cv2.waitKey(0)
