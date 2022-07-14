# ###########################################################################################
# ## This code is transfered from matlab version of the MICCAI challenge
# ## Oct 1 2019
# ###########################################################################################
# import numpy as np
# import cv2


# def is_S(mid_p_v):
#     # mid_p_v:  34 x 2
#     ll = []
#     num = mid_p_v.shape[0]
#     for i in range(num-2):
#         term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
#         term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
#         ll.append(term1-term2)
#     ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
#     ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
#     a = sum(sum(ll_pair))
#     b = sum(sum(abs(ll_pair)))
#     if abs(a-b)<1e-4:
#         return False
#     else:
#         return True

# def cobb_angle_calc(pts, image):
#     pts = np.asarray(pts, np.float32)   # 68 x 2
#     h,w,c = image.shape
#     num_pts = pts.shape[0]   # number of points, 68
#     vnum = num_pts//4-1

#     mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
#     mid_p = []
#     for i in range(0, num_pts, 4):
#         pt1 = (pts[i,:]+pts[i+2,:])/2
#         pt2 = (pts[i+1,:]+pts[i+3,:])/2
#         mid_p.append(pt1)
#         mid_p.append(pt2)
#     mid_p = np.asarray(mid_p, np.float32)   # 34 x 2

#     for pt in mid_p:
#         cv2.circle(image,
#                    (int(pt[0]), int(pt[1])),
#                    12, (0,255,255), -1, 1)

#     for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
#         cv2.line(image,
#                  (int(pt1[0]), int(pt1[1])),
#                  (int(pt2[0]), int(pt2[1])),
#                  color=(0,0,255),
#                  thickness=5, lineType=1)

#     vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
#     dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
#     mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
#     mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
#     cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
#     angles = np.arccos(cosine_angles)   # 17 x 17
#     pos1 = np.argmax(angles, axis=1)
#     maxt = np.amax(angles, axis=1)
#     pos2 = np.argmax(maxt)
#     cobb_angle1 = np.amax(maxt)
#     cobb_angle1 = cobb_angle1/np.pi*180
#     flag_s = is_S(mid_p_v)
#     if not flag_s: # not S
#         # print('Not S')
#         cobb_angle2 = angles[0, pos2]/np.pi*180
#         cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
#         cv2.line(image,
#                  (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
#                  (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
#                  color=(0, 255, 0), thickness=5, lineType=2)
#         cv2.line(image,
#                  (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
#                  (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
#                  color=(0, 255, 0), thickness=5, lineType=2)

#     else:
#         if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
#             # print('Is S: condition1')
#             angle2 = angles[pos2,:(pos2+1)]
#             cobb_angle2 = np.max(angle2)
#             pos1_1 = np.argmax(angle2)
#             cobb_angle2 = cobb_angle2/np.pi*180

#             angle3 = angles[pos2, (pos2 + 1):]
#             cobb_angle3 = np.max(angle3)
#             pos1_2 = np.argmax(angle3)
#             cobb_angle3 = cobb_angle3/np.pi*180
#             pos1_2 = pos1_2 + pos2

#             cv2.line(image,
#                      (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
#                      (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
#                      color=(0, 255, 0), thickness=5, lineType=2)

#             cv2.line(image,
#                      (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
#                      (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
#                      color=(0, 255, 0), thickness=5, lineType=2)

#         else:
#             # print('Is S: condition2')
#             angle2 = angles[pos2,:(pos2+1)]
#             cobb_angle2 = np.max(angle2)
#             pos1_1 = np.argmax(angle2)
#             cobb_angle2 = cobb_angle2/np.pi*180

#             angle3 = angles[pos2, (pos2 + 1):]
#             cobb_angle3 = np.max(angle3)
#             pos1_2 = np.argmax(angle3)
#             pos1_2 = pos1_2 + pos2
#             cobb_angle3 = cobb_angle3/np.pi*180

#             cv2.line(image,
#                      (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
#                      (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
#                      color=(0, 255, 0), thickness=5, lineType=2)

#             cv2.line(image,
#                      (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
#                      (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
#                      color=(0, 255, 0), thickness=5, lineType=2)

#     return [cobb_angle1, cobb_angle2, cobb_angle3]

###########################################################################################
## This code is transfered from matlab version of the MICCAI challenge
## Oct 1 2019
###########################################################################################
import numpy as np
import cv2
import scipy.io as sc

def curvature(angle1, angle2, angle3):
    angle = max(angle1, angle2, angle3)
    if 0 <= angle <= 10 :
        return "spinal curve"
    elif 10 < angle <= 20:
        return "mild scoliosis"
    elif 20 < angle <= 40:
        return "moderate scoliosis"
    else:
        return "severe scoliosis"

def typeOfCurve(angle2, angle3, v1, v2, apex):
    # angle2 is the thoracic (0-11)
    # angle3 is the lumbar (12-16)
    # v1 is the vertebra selected above the apex
    # v2 is the vertebra selected below the apex
    if 0 <= v1 <= 11 and 0 <= v2 <= 11:
        # double thoratic
        return "Type 5"
    elif v2 == 15 :
        # the fourth lumbar is tilted
        return "Type 4"
    elif v2 >= 12 and (angle2 >= angle3):
        return "Type 3"
    elif angle2 >= angle3 :
        # thoratic is bigger
        return "Type 2"
    elif angle2 < angle3 :
        # lumbar is bigger
        return "Type 1"
    

def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True

def cobb_angle_calc(pts, image):
    pts = np.asarray(pts, np.float32)   # 68 x 2
    print(len(pts))
    h,w,c = image.shape
    num_pts = pts.shape[0]   # number of points, 68
    # print("num_pts", num_pts)
    vnum = num_pts//4-1

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    # calculating the midway point for each vertebra
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2
    # print(len(mid_p[0]))

    # drawing a circle on the outer-edges of a mid point, Left and Right
    for pt in mid_p:
        # print("pt ", pt)
        cv2.circle(image,
                   (int(pt[0]), int(pt[1])),
                   12, (0,255,255), -1, 1)

    # connecting the lines
    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(image,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,0,255),
                 thickness=5, lineType=1)

    # cobbs angle calculation
    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    # print("vec_m", vec_m)
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    # print("dot_v", dot_v)

    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    # print("mod_v 1", mod_v)
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    # print("mod_v 2", mod_v)

    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    # print("cosine angles:", cosine_angles)
    angles = np.arccos(cosine_angles)   # 17 x 17
    # print("angles", angles)
    
    # for the ith row, the jth index with maximum angle relative to i
    pos1 = np.argmax(angles, axis=1)
    # print("pos1", pos1)
    # maximum angle of the ith vertebra
    maxt = np.amax(angles, axis=1)
    # print("maxt", maxt)
    pos2 = np.argmax(maxt)
    tempPos = pos1[pos2]
    if abs(8 - tempPos) < abs(8 - pos2):
        pos2 = tempPos
    print("first vertebra selected?", pos2)

    cobb_angle1 = np.amax(maxt)
    cobb_angle1 = cobb_angle1/np.pi*180
    print("cobb angle1 : ", cobb_angle1)
    # pos2 ~ cobb_angle1
    # pos1[pos2] = apex
    # print("cobb_angle1", cobb_angle1)

    # mid_p contains all the coordinates
    # print("coordinates:", mid_p)

    # drawing the vertebra chosen : PURPLE
    (xc0,yc0) = (int(mid_p[pos2 * 2, 0]), int(mid_p[pos2 * 2, 1]))
    (xc1,yc1) = (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1]))
    mc0 = (yc1 - yc0) / (xc1 - xc0)
    cv2.line(image,
                 (0, int(yc0 - (mc0 * xc0)) ),
                 (xc1,yc1),
                 color=(255, 0,255), thickness=5, lineType=2)

    # always draw the first vertebras line : RED
    # (xf0, yf0) = (int(mid_p[0,0]), int(mid_p[0, 1]) )
    # (xf1,yf1) = (int(mid_p[1, 0]), int(mid_p[1, 1]))
    # mf0 = int ( (yf1 - yf0) / (xf1 - xf0) )
    # cv2.line(image,
    #              (0, yf0 - (mf0 * xf0)),
    #              (xf1,yf1),
    #              color=(0, 0,255), thickness=5, lineType=2)
    # print("drew the first vertebra", xf0,yf0)
    

    flag_s = is_S(mid_p_v)

    if not flag_s: # not S
        # if the difference between ll and abs(ll) is less than 1e-4
        print('Not S')

        angle2 = angles[pos2,:(pos2+1)]
        # print("angle2", angle2)
        cobb_angle2 = np.max(angle2)
        pos1_1 = np.argmax(angle2)
        print("upper", pos1_1)
        cobb_angle2 = cobb_angle2/np.pi*180
        (x1, y1) = (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1]))
        (x2, y2) = (int(mid_p[pos1_1 * 2 + 1, 0]), int(mid_p[pos1_1 * 2 + 1, 1]))
        m1 = (y2 - y1) / (x2 - x1) 

        # angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
        angle3 = angles[pos2, (pos2 + 1):]
        cobb_angle3 = np.max(angle3)
        # print("angle3", angle3)
        pos1_2 = np.argmax(angle3)
        cobb_angle3 = cobb_angle3/np.pi*180
        # pos1_2 = pos1_2 + pos1[pos2]-1
        print("lower", pos1_2)
        pos1_2 = pos1_2 + pos2
        (x3,y3) = (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1]))
        (x4, y4) =(int(mid_p[pos1_2 * 2 + 1, 0]), int(mid_p[pos1_2 * 2 + 1, 1]))
        m2 = (y4 - y3) / (x4 - x3)
        

        # print("cobb_angle2 : ", cobb_angle2)
        # print("cobb_angle3 : ", cobb_angle3)
        
        # m1 = ( (int(mid_p[pos1_1 * 2 + 1, 1])) - int(mid_p[pos1_1 * 2, 1]) ) / (int(mid_p[pos1_1 * 2+1, 0]) - int(mid_p[pos1_1 * 2, 0]))
        cv2.line(image,
                    (0, int(y1 - (m1 * x1))),
                    (x2,y2),
                    color=(255, 0, 0), thickness=5, lineType=2)

        cv2.line(image,
                    (0, int(y3 - (m2 * x3))),
                    (x4, y4),
                    color=(255, 0, 0), thickness=5, lineType=2)

        (h1, h2) = (x2 + 10, int((y2 + yc1) / 2))
        cv2.putText(image, str(round(cobb_angle2, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 2, color = (0,0,255), thickness = 2)
        (h1, h2) = (x4 + 10, int((y3 + yc1) / 2))
        cv2.putText(image, str(round(cobb_angle3, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 2, color = (0,0,255), thickness = 2)

        curvatureSeverity = curvature(cobb_angle1, cobb_angle2, cobb_angle3)
        curvatureType = typeOfCurve(cobb_angle2, cobb_angle3, pos1_1, pos1_2, pos2)
        cv2.putText(image, curvatureSeverity + " , " + curvatureType, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 2, color = (0,0,255), thickness = 2)

    else:
        # print("calculations:", mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1], "h:", h)
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            # GREEN
            # print('Is S: condition1')
            angle2 = angles[pos2,:(pos2+1)]
            # print("angle2", angle2)
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            print("upper", pos1_1)
            cobb_angle2 = cobb_angle2/np.pi*180
            (x1, y1) = (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1]))
            (x2, y2) = (int(mid_p[pos1_1 * 2 + 1, 0]), int(mid_p[pos1_1 * 2 + 1, 1]))
            m1 = (y2 - y1) / (x2 - x1) 

            # angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            angle3 = angles[pos2, (pos2 + 1):]
            cobb_angle3 = np.max(angle3)
            # print("angle3", angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            # pos1_2 = pos1_2 + pos1[pos2]-1
            print("lower", pos1_2)
            pos1_2 = pos1_2 + pos2
            (x3,y3) = (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1]))
            (x4, y4) =(int(mid_p[pos1_2 * 2 + 1, 0]), int(mid_p[pos1_2 * 2 + 1, 1]))
            m2 = (y4 - y3) / (x4 - x3)
            

            # print("cobb_angle2 : ", cobb_angle2)
            # print("cobb_angle3 : ", cobb_angle3)
            
            # m1 = ( (int(mid_p[pos1_1 * 2 + 1, 1])) - int(mid_p[pos1_1 * 2, 1]) ) / (int(mid_p[pos1_1 * 2+1, 0]) - int(mid_p[pos1_1 * 2, 0]))
            cv2.line(image,
                     (0, int(y1 - (m1 * x1))),
                     (x2,y2),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (0, int(y3 - (m2 * x3))),
                     (x4, y4),
                     color=(0, 255, 0), thickness=5, lineType=2)
            
            # (h1, h2) = (xc1 + 10, int((yc1 + y2) / 2))
            # cv2.putText(image, str(cobb_angle1), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
            #         fontScale = 2, color = (0,0,255), thickness = 2)
            (h1, h2) = (x2 + 10, int((y2 + yc1) / 2))
            cv2.putText(image, str(round(cobb_angle2, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)
            (h1, h2) = (x4 + 10, int((y3 + yc1) / 2))
            cv2.putText(image, str(round(cobb_angle3, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)

            curvatureSeverity = curvature(cobb_angle1, cobb_angle2, cobb_angle3)
            curvatureType = typeOfCurve(cobb_angle2, cobb_angle3, pos1_1, pos1_2, pos2)
            cv2.putText(image, curvatureSeverity + " , " + curvatureType, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)
        else:
            # white
            # print('Is S: condition2')
            angle2 = angles[pos2,:(pos2+1)]
            # print("angle2", len(angle2), angle2)
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            print("largest tilt for upper", pos1_1)
            cobb_angle2 = cobb_angle2/np.pi*180
            
            (x1, y1) = (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1]))
            (x2, y2) = (int(mid_p[pos1_1 * 2 + 1, 0]), int(mid_p[pos1_1 * 2 + 1, 1]))
            m1 = (y2 - y1) / (x2 - x1) 
            # print(x1,y1)

            # angle3 = angles[pos1_1, :(pos1_1+1)]
            angle3 = angles[pos2, (pos2 + 1):]
            # print("angle3", angle3)
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            print("largest tilt for lower", pos1_2)
            cobb_angle3 = cobb_angle3/np.pi*180
            # pos1_2 = pos1_2 + pos1[pos2]-1
            pos1_2 = pos1_2 + pos2
            # print("new pos1_2", pos1_2)


            # print("cobb_angle2 : ", cobb_angle2)
            # print("cobb_angle3 : ", cobb_angle3)

            (x3,y3) = (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1]))
            (x4, y4) =(int(mid_p[pos1_2 * 2 + 1, 0]), int(mid_p[pos1_2 * 2 + 1, 1]))
            m2 = (y4 - y3) / (x4 - x3)
            # m = ( (int(mid_p[pos1_1 + 1, 1])) - int(mid_p[pos1_1, 1]) ) / (int(mid_p[pos1_1+1, 0]) - int(mid_p[pos1_1, 0]))
            cv2.line(image,
                     (0, int(y1 - (m1 * x1))),
                     (x2, y2),
                     color=(255, 255, 255), thickness=5, lineType=2)
# y = mx + b => b = y - mx
            cv2.line(image,
                     (0, int(y3 - (m2 * x3))),
                     (x4, y4),
                     color=(255, 255, 255), thickness=5, lineType=2)

            # (h1, h2) = (xc1 + 10, int((yc1 + y2) / 2))
            # cv2.putText(image, str(cobb_angle1), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
            #         fontScale = 2, color = (0,0,255), thickness = 2)
            (h1, h2) = (x2 + 10, int((y2 + yc1) / 2))
            cv2.putText(image, str(round(cobb_angle2, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)
            (h1, h2) = (x4 + 10, int((y3 + yc1) / 2))
            cv2.putText(image, str(round(cobb_angle3, 3)), (h1,h2), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)

            curvatureSeverity = curvature(cobb_angle1, cobb_angle2, cobb_angle3)
            curvatureType = typeOfCurve(cobb_angle2, cobb_angle3, pos1_1, pos1_2, pos2)
            cv2.putText(image, curvatureSeverity + " " + curvatureType, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale = 2, color = (0,0,255), thickness = 2)
    return [cobb_angle1, cobb_angle2, cobb_angle3]
    # return angles

# trying to display stuff
# done = False
# while not done:
#     try:
#         imageName = input("Enter Image Name: ").lower()
#         if imageName == 'done':
#             break
#         img = cv2.imread(imageName + '.jpg')

#         path = '/Users/mc/Desktop/local-ML/practice/'
#     # handle exceptions?
#         pts0 = sc.loadmat(path + imageName + '.mat')
#         pts1 = pts0['p2']
#         # cv2.imshow("without angles", img)
#         angles = cobb_angle_calc(pts1, img)
#         # print("cobbs angles:", angles)
#         # markedImg = cv2.putText(img, angles, (150.150), cv2.FONT_HERSHEY_SIMPLEX, 
#         #            1, (0,255,0), 2)
#         # cv2.imwrite(path + 'pillar_text.jpg', co)
#         cv2.imshow('cobbs angles', img)
       
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#     except:
#         print('image not found, try again')
#         pass






