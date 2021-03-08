# First part copy pasted from the "Machine learning and Datamining" course, exercise 2
from get_subjects import *
import matplotlib.pyplot as plt
from scipy.linalg import svd
import random

# set random seed
seed = 1510
np.random.seed(seed)
random.seed(seed)

# Number of samples
N = 1000

# THIS MIGHT TAKE SOME TIME
#selected = get_fnames( ["dance", "walk"] )
#data = parse_selected( selected )
#X, y = gather_all_np(data)

#Load saved numpy arrays (you'll have to run get_subjects once, before this works)
X = np.load('X_dance_np.npy')
y = np.load('y_dance_np.npy')

idx = np.random.choice(X.shape[0], N, replace=False)

X_sample = X[idx, :(X.shape[1]-3)]
y_sample = y[idx]
classNames, factors = np.unique(y_sample, return_inverse=True)
C = len(classNames)

# Subtract mean value from data
X_norm = (X_sample - X_sample.mean(0)) / X_sample.std(0)

# PCA by computing SVD of Y
U,S,Vh = svd(X_norm,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Project the centered data onto principal component space
Z = X_norm @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title("MoCap data: PCA")
#Z = array(Z)
for c in list(range(C)):
    # select indices belonging to class c:
    class_mask = factors==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()

# --------VARIANCE EXPLAINED---------
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9
# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#### PLOT OF ATTRIBUTES
plt.figure(figsize=(10,15))
# Plot attribute coefficients in principal component space
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    #plt.text(V[att,i], V[att,j], attributeNames[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Zero-mean and unit variance - Attribute coefficients')
plt.axis('equal')
plt.show()
