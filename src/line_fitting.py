import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn import linear_model, datasets


n_samples = 100
n_outliers = 20
th=30*np.pi/180
w=2
l=4.5
x_off = random.uniform(10, 35)
y_off = random.uniform(10,20)
x_off=25 #depth
yoff= 10
random_l =np.random.uniform(0,l,n_samples)
random_w = np.random.uniform(0,w,n_outliers)

x1= x_off + random_l*np.cos(th)
y1= y_off+ random_l * np.sin(th)

x2=x_off - random_w*np.sin(th)
y2= y_off + random_w*np.cos(th)
#
# x2=x_off + l * np.cos(th)
# y2= y_off + l*np.sin(th)
X=x1
y=y1
X[:n_outliers] = x2
y[:n_outliers] = y2

X=X.reshape(-1, 1)
# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
#                                       n_informative=1, noise=10,
                                      # coef=True, random_state=0)
# linspace()
# Add outlier data
# np.random.seed(0)
# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.linspace(X.min(), X.max(),10)[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)
m1 = ransac.estimator_.coef_[0]
b1 = ransac.estimator_.intercept_
# Compare estimated coefficients
if not all(inlier_mask) and sum(outlier_mask)>10:

    X2 = X[outlier_mask, :]
    y2 = y[outlier_mask]
    ransac2 = linear_model.RANSACRegressor()
    ransac2.fit(X2, y2)
    m2 = ransac2.estimator_.coef_[0]
    b2 = ransac2.estimator_.intercept_
    A = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([b1, b2])
    corner = np.linalg.solve(A, b)
    x1mid = (np.vstack((X[inlier_mask], corner[0])).min() + np.vstack((X[inlier_mask], corner[0])).max()) / 2
    y1mid = ransac.predict(x1mid.reshape(1,-1))[0]
    x2mid = (np.vstack((X2, corner[0])).min() + np.vstack((X2, corner[0])).max()) / 2
    y2mid = ransac2.predict(x2mid.reshape(1,-1))[0]
    line_X2 = np.linspace(X2.min(), X2.max(), 100)[:, np.newaxis]
    line_y_ransac2 = ransac2.predict(line_X2)
    Amid=np.array([[1/m1,1],[1/m2,1]])
    bmid=np.array([y1mid+x1mid*1/m1, y2mid+x2mid*1/m2])
    u1=np.array([x1mid, y1mid])-corner
    d1 = np.linalg.norm(u1)*2
    u2=np.array([x2mid,y2mid])-corner
    d2= np.linalg.norm(u2)*2
    pos_c = np.linalg.solve(Amid, bmid)
    theta1 = np.arctan2(u1[1],u1[0])
    theta2 = np.arctan2(u2[1],u1[0])

    # if d1>d2:
    #     #means d1 is the longer side
    #     return [corner, d1, u1, d2, u2, pos_c, theta1]
    # if d2>d1:
    #     #meand d2 is the longer side
    #     return [corner, d2, u2, d1, u1, pos_c]
else:

    x1mid = (X[inlier_mask].min() + X[inlier_mask].max() )/ 2
    y1mid = ransac.predict(x1mid.reshape(1, -1))[0]
    xmin=X[inlier_mask].min()
    ymin=ransac.predict(np.array(xmin).reshape(1,-1))
    xmax=X[inlier_mask].max()
    ymax=ransac.predict(np.array(xmax).reshape(1,-1))
    u1=np.array([xmin-xmax,ymin-ymax])
    d1=np.linal.norm(u1)

    print(d1)
    if d1 >2.75:
        lout = d1
        theta = np.arctan2(u1[1],u1[0])
    else:
        wout = np.arctan2()





print("Estimated coefficients (linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_, ransac.estimator_.intercept_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.scatter(x1mid, y1mid, color='green')
try:
    plt.plot(line_X2, line_y_ransac2, color='red', linewidth=lw, label='RANSAC regressor')

    plt.scatter(x2mid, y2mid, color='green')
    plt.scatter(pos_c[0], pos_c[1], color='green')
except:
    ('No outliers')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")

plt.axis('equal')
plt.xlim([0,40])
plt.ylim([0,30])

plt.show()