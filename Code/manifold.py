import numpy as np
from numpy.linalg import norm
import utils
from sklearn.decomposition import PCA
from findMin import findMin

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        pca = PCA(k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z, f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()



class ISOMAP(MDS):

    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.nn = n_neighbours

    def compress(self, X):
        n = X.shape[0]
        k = self.k
        nn = self.nn

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        G = np.zeros([n,n])

        for i in range(n):
            sorted_D = np.sort(D[:,i])
            sorted_I = np.argsort(D[:,i])

            d = sorted_D[nn]

            neighs = sorted_I[nn]

            G[neighs, i] = d
            G[i, neighs] = d


        D = np.zeros([n,n])
        maxD = 0

        for i in range(n):
            for j in range(n):
                if j != i:
                    cost = utils.dijkstra(G, i, j)
                    if (not np.isinf(cost)) & (cost > maxD):
                        maxD = cost

                    D[i,j] = cost


        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.
        D[np.isinf(D)] = D[~np.isinf(D)].max()

        # Initialize low-dimensional representation with PCA
        pca = PCA(self.k)
        pca.fit(X)
        Z = pca.transform(X)

        # Solve for the minimizer
        z,f = findMin(self._fun_obj_z, Z.flatten(), 500, D)
        Z = z.reshape(n, self.k)
        return Z
