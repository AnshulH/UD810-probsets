import cv2
import numpy as np 

# incomplete

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

def hough_peaks(Acc,peaks=1,thresh=100,nhood=5):
	peak = np.zeros((peaks,2), dtype=np.uint64)
	temp = Acc.copy()
	for i in range(peaks):
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp)
		if max_val > thresh:
			peak[i] = max_loc
			(c,r) = max_loc
			t = nhood//2.0
			temp[int(max(r-t,0)):int(max(r+t+1,0)),int(max(c-t,0)):int(c+t+1)] = 0
		else:
			peak = peak[:i]
			break
		return peak[:,::-1]	

img = cv2.imread('ps1-input0.png', cv2.IMREAD_GRAYSCALE)
edged = cv2.Canny(img, 100, 200)
Acc, thetas, rhos = hough_acc(img)
peaks = hough_peaks(Acc,peaks=10)

temp_peak = Acc.copy()

for peak in peaks:
	cv2.circle(temp_peak, tuple(peak[::-1]), 5, (255,255,255), -1)

cv2.imshow('img',temp_peak)
#cv2.imwrite('temph.png',img1)


#cv2.imwrite('output/ps1-2-a-1.png', Acc)
