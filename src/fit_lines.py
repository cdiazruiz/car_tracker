import numpy as np
from sklearn import linear_model, datasets


def lines(X,y):
    X=X.reshape(-1,1)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.linspace(X.min(), X.max(), 10)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    m1 = ransac.estimator_.coef_[0]
    b1 = ransac.estimator_.intercept_
    # Compare estimated coefficients
    if not all(inlier_mask) and sum(outlier_mask) > 10:

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
        y1mid = ransac.predict(x1mid.reshape(1, -1))[0]
        x2mid = (np.vstack((X2, corner[0])).min() + np.vstack((X2, corner[0])).max()) / 2
        y2mid = ransac2.predict(x2mid.reshape(1, -1))[0]
        line_X2 = np.linspace(X2.min(), X2.max(), 100)[:, np.newaxis]
        line_y_ransac2 = ransac2.predict(line_X2)
        Amid = np.array([[1 / m1, 1], [1 / m2, 1]])
        bmid = np.array([y1mid + x1mid * 1 / m1, y2mid + x2mid * 1 / m2])
        u1 = np.array([x1mid, y1mid]) - corner
        d1 = np.linalg.norm(u1) * 2
        u2 = np.array([x2mid, y2mid]) - corner
        d2 = np.linalg.norm(u2) * 2
        pos_c = np.linalg.solve(Amid, bmid)
        theta1 = np.arctan2(u1[1], u1[0])
        theta2 = np.arctan2(u2[1], u1[0])

        if d1>d2:
            #means d1 is the longer side
            return [corner, d1, u1, d2, u2, pos_c, theta1,0]
        if d2>d1:
            #meand d2 is the longer side
            return [corner, d2, u2, d1, u1, pos_c, theta2, 0]

    else:

        x1mid = (X[inlier_mask].min() + X[inlier_mask].max()) / 2
        y1mid = ransac.predict(x1mid.reshape(1, -1))[0]
        xmin = X[inlier_mask].min()
        ymin = ransac.predict(np.array(xmin).reshape(1, -1))
        xmax = X[inlier_mask].max()
        posmin = np.array([xmin,ymin])
        ymax = ransac.predict(np.array(xmax).reshape(1, -1))
        u1 = np.array([xmax - xmin, ymax - ymin])
        d1 = np.linalg.norm(u1)
        midpoint=np.array([x1mid, y1mid])

        if d1 > 2.75:
            lout = d1
            theta = np.arctan2(u1[1], u1[0])
            return [midpoint,lout,u1, posmin, theta, 1]

        else:
            wout = d1
            theta = np.arctan2(u1[1], u1[0])
            return [midpoint,wout,u1,posmin, theta, 2]