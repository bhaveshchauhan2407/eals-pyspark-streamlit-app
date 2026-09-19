from typing import Dict, List, Tuple
import numpy as np

from src.config import FastALSConfig
from src.predict import predict_score


class FastALSModel:
    """
    Beginner-friendly Python skeleton of the Java MF_fastALS class.

    This version keeps the structure of the Java code,
    but does not yet implement full PySpark/RDD training.
    """

    def __init__(
        self,
        interactions: List[Tuple[int, int]],
        config: FastALSConfig,
    ):
        self.config = config
        self.interactions = interactions

        # Build sorted user/item index sets
        self.user_ids = sorted({u for u, _ in interactions})
        self.item_ids = sorted({i for _, i in interactions})

        self.user_count = len(self.user_ids)
        self.item_count = len(self.item_ids)

        # Map raw IDs to matrix row/column indices
        self.user_to_index = {u: idx for idx, u in enumerate(self.user_ids)}
        self.item_to_index = {i: idx for idx, i in enumerate(self.item_ids)}

        self.index_to_user = {idx: u for u, idx in self.user_to_index.items()}
        self.index_to_item = {idx: i for i, idx in self.item_to_index.items()}

        # Training matrix structures (Python replacements for SparseMatrix access)
        self.user_items = self._build_user_items()
        self.item_users = self._build_item_users()

        # Positive weights W(u, i)
        self.W = self._build_positive_weights()

        # Negative item weights Wi[i]
        self.Wi = self._compute_item_missing_weights()

        # Model parameters U and V
        self.U = None
        self.V = None

        # Caches SU and SV
        self.SU = None
        self.SV = None

        # Helper arrays used in Java code
        self.prediction_users = np.zeros(self.user_count)
        self.prediction_items = np.zeros(self.item_count)
        self.rating_users = np.zeros(self.user_count)
        self.rating_items = np.zeros(self.item_count)
        self.w_users = np.zeros(self.user_count)
        self.w_items = np.zeros(self.item_count)

        self._initialize_factors()
        self._init_caches()

    def _build_user_items(self) -> Dict[int, List[int]]:
        user_items = {u_idx: [] for u_idx in range(self.user_count)}
        for raw_u, raw_i in self.interactions:
            u = self.user_to_index[raw_u]
            i = self.item_to_index[raw_i]
            user_items[u].append(i)
        return user_items

    def _build_item_users(self) -> Dict[int, List[int]]:
        item_users = {i_idx: [] for i_idx in range(self.item_count)}
        for raw_u, raw_i in self.interactions:
            u = self.user_to_index[raw_u]
            i = self.item_to_index[raw_i]
            item_users[i].append(u)
        return item_users

    def _build_positive_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Java sets positive interaction weights to 1 by default.
        """
        weights = {}
        for raw_u, raw_i in self.interactions:
            u = self.user_to_index[raw_u]
            i = self.item_to_index[raw_i]
            weights[(u, i)] = 1.0
        return weights

    def _compute_item_missing_weights(self) -> np.ndarray:
        """
        Python equivalent of Wi computation from the Java constructor.
        """
        item_popularity = np.zeros(self.item_count)

        for i in range(self.item_count):
            item_popularity[i] = len(self.item_users[i])

        total = item_popularity.sum()
        if total == 0:
            return np.zeros(self.item_count)

        probabilities = item_popularity / total
        powered = np.power(probabilities, self.config.alpha)

        z = powered.sum()
        if z == 0:
            return np.zeros(self.item_count)

        wi = self.config.w0 * powered / z
        return wi

    def _initialize_factors(self) -> None:
        """
        Initialize U and V like Java DenseMatrix.init(mean, stdev).
        """
        np.random.seed(self.config.random_seed)

        self.U = np.random.normal(
            loc=self.config.init_mean,
            scale=self.config.init_stdev,
            size=(self.user_count, self.config.factors),
        )

        self.V = np.random.normal(
            loc=self.config.init_mean,
            scale=self.config.init_stdev,
            size=(self.item_count, self.config.factors),
        )

    def _init_caches(self) -> None:
        """
        Equivalent of Java initS():
        SU = U^T U
        SV = V^T * diag(Wi) * V
        """
        self.SU = self.U.T @ self.U

        weighted_V = self.V * self.Wi[:, np.newaxis]
        self.SV = self.V.T @ weighted_V

    def predict(self, u: int, i: int) -> float:
        return predict_score(self.U[u], self.V[i])

    def set_train(self, interactions: List[Tuple[int, int]]) -> None:
        """
        Python placeholder version of setTrain(...)
        """
        self.interactions = interactions
        self.user_items = self._build_user_items()
        self.item_users = self._build_item_users()
        self.W = self._build_positive_weights()

    def set_uv(self, U: np.ndarray, V: np.ndarray) -> None:
        """
        Python version of setUV(...)
        """
        self.U = U.copy()
        self.V = V.copy()
        self._init_caches()

    def _add_new_user(self, raw_user_id: int) -> int:
        new_index = self.user_count

        self.user_to_index[raw_user_id] = new_index
        self.index_to_user[new_index] = raw_user_id
        self.user_ids.append(raw_user_id)

        self.user_items[new_index] = []

        new_row = np.random.normal(
            loc=self.config.init_mean,
            scale=self.config.init_stdev,
            size=(1, self.config.factors),
        )
        self.U = np.vstack([self.U, new_row])

        self.prediction_users = np.append(self.prediction_users, 0.0)
        self.rating_users = np.append(self.rating_users, 0.0)
        self.w_users = np.append(self.w_users, 0.0)

        self.user_count += 1
        self.SU = self.U.T @ self.U
        return new_index


    def _add_new_item(self, raw_item_id: int) -> int:
        new_index = self.item_count

        self.item_to_index[raw_item_id] = new_index
        self.index_to_item[new_index] = raw_item_id
        self.item_ids.append(raw_item_id)

        self.item_users[new_index] = []

        new_row = np.random.normal(
            loc=self.config.init_mean,
            scale=self.config.init_stdev,
            size=(1, self.config.factors),
        )
        self.V = np.vstack([self.V, new_row])

        self.prediction_items = np.append(self.prediction_items, 0.0)
        self.rating_items = np.append(self.rating_items, 0.0)
        self.w_items = np.append(self.w_items, 0.0)

        self.Wi = np.append(self.Wi, self.config.w0 / max(1, self.item_count + 1))

        self.item_count += 1
        self._init_caches()
        return new_index


    def update_model(self, raw_user_id: int, raw_item_id: int, w_new: float = 4.0, online_iter: int = 1):
        """
        Python approximation of Java updateModel(u, i).
        """
        if raw_user_id not in self.user_to_index:
            u = self._add_new_user(raw_user_id)
        else:
            u = self.user_to_index[raw_user_id]

        if raw_item_id not in self.item_to_index:
            i = self._add_new_item(raw_item_id)
        else:
            i = self.item_to_index[raw_item_id]

        if i not in self.user_items[u]:
            self.user_items[u].append(i)
        if u not in self.item_users[i]:
            self.item_users[i].append(u)

        self.W[(u, i)] = w_new

        for _ in range(online_iter):
            self.update_user(u)
            self.update_item(i)

    def loss(self):
        """
        Python version of the fast loss computation in the Java code.
        """
        reg_loss = self.config.reg * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
        total_loss = reg_loss

        for u in range(self.user_count):
            l = 0.0
            user_vector = self.U[u]

            for i in self.user_items[u]:
                pred = self.predict(u, i)
                rating_ui = 1.0
                weight_ui = self.W[(u, i)]
                l += weight_ui * (rating_ui - pred) ** 2
                l -= self.Wi[i] * (pred ** 2)

            l += user_vector @ self.SV @ user_vector
            total_loss += l

        return float(total_loss)

    def update_user(self, u: int) -> None:
        item_list = self.user_items[u]
        if len(item_list) == 0:
            return

        for i in item_list:
            self.prediction_items[i] = self.predict(u, i)
            self.rating_items[i] = 1.0
            self.w_items[i] = self.W[(u, i)]

        old_vector = self.U[u].copy()

        for f in range(self.config.factors):
            numer = 0.0
            denom = 0.0

            for k in range(self.config.factors):
                if k != f:
                    numer -= self.U[u, k] * self.SV[f, k]

            for i in item_list:
                self.prediction_items[i] -= self.U[u, f] * self.V[i, f]
                numer += (
                    self.w_items[i] * self.rating_items[i]
                    - (self.w_items[i] - self.Wi[i]) * self.prediction_items[i]
                ) * self.V[i, f]
                denom += (self.w_items[i] - self.Wi[i]) * (self.V[i, f] ** 2)

            denom += self.SV[f, f] + self.config.reg

            if denom != 0 and np.isfinite(numer) and np.isfinite(denom):
                new_value = numer / denom
                if np.isfinite(new_value):
                    self.U[u, f] = np.clip(new_value, -10.0, 10.0)

            for i in item_list:
                self.prediction_items[i] += self.U[u, f] * self.V[i, f]

        for f in range(self.config.factors):
            for k in range(f + 1):
                val = self.SU[f, k] - old_vector[f] * old_vector[k] + self.U[u, f] * self.U[u, k]
                self.SU[f, k] = val
                self.SU[k, f] = val

    def update_item(self, i: int) -> None:
        user_list = self.item_users[i]
        if len(user_list) == 0:
            return

        for u in user_list:
            self.prediction_users[u] = self.predict(u, i)
            self.rating_users[u] = 1.0
            self.w_users[u] = self.W[(u, i)]

        old_vector = self.V[i].copy()

        for f in range(self.config.factors):
            numer = 0.0
            denom = 0.0

            for k in range(self.config.factors):
                if k != f:
                    numer -= self.V[i, k] * self.SU[f, k]
            numer *= self.Wi[i]

            for u in user_list:
                self.prediction_users[u] -= self.U[u, f] * self.V[i, f]
                numer += (
                    self.w_users[u] * self.rating_users[u]
                    - (self.w_users[u] - self.Wi[i]) * self.prediction_users[u]
                ) * self.U[u, f]
                denom += (self.w_users[u] - self.Wi[i]) * (self.U[u, f] ** 2)

            denom += self.Wi[i] * self.SU[f, f] + self.config.reg

            if denom != 0 and np.isfinite(numer) and np.isfinite(denom):
                new_value = numer / denom
                if np.isfinite(new_value):
                    self.V[i, f] = np.clip(new_value, -10.0, 10.0)

            for u in user_list:
                self.prediction_users[u] += self.U[u, f] * self.V[i, f]

        for f in range(self.config.factors):
            for k in range(f + 1):
                val = (
                    self.SV[f, k]
                    - old_vector[f] * old_vector[k] * self.Wi[i]
                    + self.V[i, f] * self.V[i, k] * self.Wi[i]
                )
                self.SV[f, k] = val
                self.SV[k, f] = val