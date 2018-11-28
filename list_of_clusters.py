"""Algoritmo de clustering List of Clusters"""

# Author: Juan Martín Loyola <jmloyola@outlook.com>

import numpy as np
from scipy.spatial import distance

np.random.seed(0)

def get_distance(X, D, i, j, distance_metric='euclidean'):
    if D[i,j] == -1:
        D[i,j] = distance.cdist(X[i].reshape(1,X.shape[1]),
                                X[j].reshape(1,X.shape[1]),
                                distance_metric)
        D[j,i] = D[i,j]
    return D[i,j]

def build_list_clusters(X, center_choise, fixed_radius,
                        fixed_size, distance_metric):
    """Algoritmo de clustering List of Clusters.

    Parámetros
    ----------
    X : arreglo, tamaño (n_samples, n_features)
        Las observaciones a clusterizar.

    center_choise : {'p1', 'p2', 'p3', 'p4', 'p5'}, opcional,
                    por defecto: 'p1'
        Heurística de selección del centro:
            - 'p1': aleatoria
            - 'p2': el elemento más cercano al centro anterior
                    en el conjunto restante
            - 'p3': el elemento más alejado al centro anterior
                    en el conjunto restante
            - 'p4': el elemento que minimice la suma de las
                    distancia a los centros previos
            - 'p5': el elemento que maximice la suma de las
                    distancia a los centros previos

    fixed_radius : float, opcional, por defecto: None
        La longitud de radio de cada cluster.

    fixed_size : int, opcional, por defecto: None
        El número de elementos en cada cluster.

    distance_metric : {'euclidean', 'cityblock'}, opcional,
                    por defecto: 'euclidean'
        La medida de distancia a considerar.

    Retorna
    -------
    centroid : float ndarray
        centros encontrados.

    radius : integer ndarray
        radius[i] es el radio del i'esimo centroide.

    label : integer ndarray con tamaño (n_samples,)
        label[i] es el codigo o indice del centroide del cluster al
        que elemento i'esimo pertenece.

    """
    n_samples, n_features = X.shape
    D = -np.ones((n_samples, n_samples))

    centers = []
    centers_idx = []
    radius = []
    labels = -np.ones((n_samples,))

    label_counter = 0

    elementos_restantes = list(range(n_samples))

    while elementos_restantes:
        if center_choise == 'p1':
            c = np.random.choice(elementos_restantes, size=1)
        elif center_choise == 'p2':
            if centers == []:
                c = np.random.choice(elementos_restantes, size=1)
            else:
                last_center_idx = centers_idx[-1]
                min_dist = np.inf
                idx_min_dist = None
                for r in elementos_restantes:
                    dist = get_distance(X, D, last_center_idx, r,
                                        distance_metric)
                    if dist < min_dist:
                        min_dist = dist
                        idx_min_dist = r
                c = idx_min_dist
        elif center_choise == 'p3':
            if centers == []:
                c = np.random.choice(elementos_restantes, size=1)
            else:
                last_center_idx = centers_idx[-1]
                max_dist = -np.inf
                idx_max_dist = None
                for r in elementos_restantes:
                    dist = get_distance(X, D, last_center_idx, r,
                                        distance_metric)
                    if max_dist < dist:
                        max_dist = dist
                        idx_max_dist = r
                c = idx_max_dist
        elif center_choise == 'p4':
            if centers == []:
                c = np.random.choice(elementos_restantes, size=1)
            else:
                min_dist = np.inf
                idx_min_dist = None
                for r in elementos_restantes:
                    suma_dist = 0
                    for x in centers_idx:
                        suma_dist += get_distance(X, D, r, x, distance_metric)
                    if suma_dist < min_dist:
                        min_dist = suma_dist
                        idx_min_dist = r
                c = idx_min_dist
        elif center_choise == 'p5':
            if centers == []:
                c = np.random.choice(elementos_restantes, size=1)
            else:
                max_dist = -np.inf
                idx_max_dist = None
                for r in elementos_restantes:
                    suma_dist = 0
                    for x in centers_idx:
                        suma_dist += get_distance(X, D, r, x, distance_metric)
                    if max_dist < suma_dist:
                        max_dist = suma_dist
                        idx_max_dist = r
                c = idx_max_dist

        # Transformamos en vector (n_features,) la matriz de dos
        # dimensiones (1, n_features).
        np_c = X[c].flatten()
        # Agregamos el centro actual a la lista de centros.
        centers.append([x for x in np_c])
        centers_idx.append(c)

        elementos_restantes.pop(elementos_restantes.index(c))

        labels[c] = label_counter

        if fixed_size is None:
            for r in list(elementos_restantes):
                dist = get_distance(X, D, c, r, distance_metric)
                if dist < fixed_radius:
                    labels[r] = label_counter
                    elementos_restantes.pop(elementos_restantes.index(r))
            radius.append(fixed_radius)
        else:
            for r in elementos_restantes:
                _ = get_distance(X, D, c, r, distance_metric)
            sorted_idx = np.argsort(D[elementos_restantes,c])
            in_cluster = []
            for j in sorted_idx[:fixed_size].tolist():
                in_cluster.append(elementos_restantes[j])
            max_distance = get_distance(X, D, c, in_cluster[-1], distance_metric)
            labels[in_cluster] = label_counter
            for x in in_cluster:
                elementos_restantes.pop(elementos_restantes.index(x))
            radius.append(max_distance)

        label_counter += 1

    return np.array(centers), np.array(radius), labels


class ListOfClusters():
    """Algoritmo de clustering ListOfClusters

    Parámetros
    ----------
    center_choise : {'p1', 'p2', 'p3', 'p4', 'p5'}, opcional, por defecto: 'p1'
        Heurística de selección del centro:
            - 'p1': aleatoria
            - 'p2': el elemento más cercano al centro anterior
                    en el conjunto restante
            - 'p3': el elemento más alejado al centro anterior
                    en el conjunto restante
            - 'p4': el elemento que minimice la suma de las
                    distancia a los centros previos
            - 'p5': el elemento que maximice la suma de las
                    distancia a los centros previos

    fixed_radius : float, opcional, por defecto: None
        La longitud de radio de cada cluster.

    fixed_size : int, opcional, por defecto: None
        El número de elementos en cada cluster.

    distance_metric : {'euclidean', 'cityblock'}, opcional,
                    por defecto: 'euclidean'
        La medida de distancia a considerar.

    Atributos
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordenadas a los centros de los clusters.

    cluster_radius_ : array, (n_clusters,)
        Radios de los clusters.

    labels_ :
        Categoria de cada punto.

    Ejemplo
    --------

    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> lc = ListOfClusters().fit(X)
    >>> lc.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> lc.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])

    """

    def __init__(self, center_choise='p1', fixed_radius=1.0,
                 fixed_size=None, distance_metric='euclidean'):

        self.center_choise = center_choise
        self.fixed_radius = fixed_radius
        self.fixed_size = fixed_size
        self.distance_metric = distance_metric

    def fit(self, X, y=None):
        """Cálculo de la Lista de Clusters.

        Parámetros
        ----------
        X : arreglo, tamaño (n_samples, n_features)
            Las observaciones a clusterizar.

        y : Ignorado

        """
        self.cluster_centers_, self.cluster_radius_, self.labels_ = \
            build_list_clusters(
                X, center_choise=self.center_choise,
                fixed_radius=self.fixed_radius,
                fixed_size=self.fixed_size,
                distance_metric=self.distance_metric)
        return self

    def fit_predict(self, X, y=None):
        """Cálculo de los centros de clusters y predicción del índice
        de cluster para cada instancia.

        Método conveniente; equivalente a llamar a fit(X) seguido
        de predict(X).

        Parámetros
        ----------
        X : arreglo, tamaño (n_samples, n_features)
            Nueva data para transformar.

        u : Ignorado

        Retorna
        -------
        labels : array, tamaño [n_samples,]
            Índice de los clusters a los que pertenece cada instancia.
        """
        return self.fit(X).labels_
