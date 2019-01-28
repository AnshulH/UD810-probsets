import cv2
import numpy as np 

def hough_acc(img,rho_res=1):

	height, width = img.shape
	diag = np.sqrt(height**2 + width**2)
	rhos = np.arange(-diag,diag+1,1)
	thetas = np.deg2rad(np.arange(-90,90,1))

	acc = np.zeros((len(rhos),len(thetas)),dtype=np.uint64)
	y_idx, x_idx = np.nonzero(img)

	for i in range(len(x_idx)):
		x = x_idx[i]
		y = y_idx[i]

		for j in range(len(thetas)):
			rho = int((x*np.cos(thetas[j]) + y*np.sin(thetas[j]) + diag))//rho_res
			acc[rho,j] += 1

	return acc, thetas, rhos

def hough_peaks(H, num_peaks=5, nhood_size=3):
    
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)

        
        idx_y, idx_x = H1_idx 
 
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = int(idx_x - (nhood_size/2))
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = int(H.shape[1])
        else: max_x = int(idx_x + (nhood_size/2) + 1)
		
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = int(idx_y - (nhood_size/2))
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = int(H.shape[0])
        else: max_y = int(idx_y + (nhood_size/2) + 1)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                
                H1[y, x] = 0

                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    return indicies, H	


img = cv2.imread('ps1-input0.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('ps1-input0.png')
edged = cv2.Canny(img, 100, 200)
Acc, thetas, rhos = hough_acc(edged)
indicies, H = hough_peaks(Acc,num_peaks=8)

cv2.imwrite('temp.png', H)

for i in range(len(indicies)):
        
    rho = rhos[indicies[i][0]]
    theta = thetas[indicies[i][1]]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 500*(-b))
    y1 = int(y0 + 500*(a))
    x2 = int(x0 - 500*(-b))
    y2 = int(y0 - 500*(a))

    cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 255), 2)

cv2.imwrite('detect_lines.png', img1)
#cv2.imwrite('output/ps1-2-a-1.png', Acc)
