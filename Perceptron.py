# Import library
import numpy as np
import matplotlib.pyplot as plt


# Buat kelas Perceptron
class Perceptron:
    # Simpan learning rate dan max epoch dalam konstruktor
    def __init__(self, alpha=0.1, epoch=10, show_plot=True):
        self.alpha = alpha
        self.epoch = epoch
        self.show_plot = show_plot

    # Fungsi menghitung nilai y_in atau net
    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Fungsi menerapkan fungsi aktivasi bipolar
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    # Fungsi membuat simulasi garis pemisah data
    def plot_decision_boundary(self, X, t, epoch):
        if not self.show_plot:
            return

        plt.figure()

        # Membuat titik data input
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=t.ravel(),
            marker="o",
            edgecolors="k",
            cmap=plt.cm.RdYlBu,
        )

        # Menentukan limit tampilan bidang grafik
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Membuat garis pemisah
        x_vals = np.linspace(x_min, x_max, 100)
        if self.w_[2] != 0:
            y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]
            plt.plot(
                x_vals,
                y_vals,
                "b-",
                label=f"Decision boundary (Epoch {epoch + 1})",
            )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch + 1}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)

    # Fungsi utama Perceptron
    def fit(self, X, t):
        # Inisialisasi bobot dan bias awal = 0
        self.w_ = np.zeros(1 + X.shape[1])

        # Menyimpan hasil pada HasilPerceptron.txt
        with open("HasilPerceptron.txt", "w", encoding="utf-8") as f:
            f.write("Masalah OR dengan Perceptron\n")
            f.write("----------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target:\n{t}\n")
            f.write(f"Bobot awal : {self.w_[1:]}\n")
            f.write(f"Bias awal : {self.w_[0]}\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n")

            # Iterasi Perceptron (Epoch)
            for epoch in range(self.epoch):
                f.write(f"\nEpoch {epoch + 1}/{self.epoch}\n")
                f.write("----------\n")
                error = np.array([])

                # Iterasi setiap pasangan input dan target
                for xi, target in zip(X, t):
                    target = float(np.ravel(target)[0])

                    # Prediksi
                    y_pred = float(self.predict(xi))

                    # Hitung error
                    err = target - y_pred
                    error = np.append(error, err)

                    # Delta rule
                    update = self.alpha * err

                    # Update bobot dan bias
                    self.w_[1:] += update * xi
                    self.w_[0] += update

                    # Simpan hasil
                    f.write(
                        f"Input: {xi}, "
                        f"Target: {target}, "
                        f"Predict: {y_pred}, "
                        f"Error: {err}, "
                        f"Bobot: {self.w_[1:]}, "
                        f"Bias: {self.w_[0]}\n"
                    )

                # Simulasikan garis pemisah
                f.flush()
                self.plot_decision_boundary(X, t, epoch)

                # Hitung SSE
                sse = np.sum(error**2)
                f.write(f"Sum Square Error(SSE): {sse}\n")

                # Kondisi berhenti
                if sse == 0 or epoch + 1 == self.epoch:
                    f.write("----------------------------------------\n")
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena ")

                    if sse == 0:
                        f.write("Sum Square Error(SSE) mencapai target.\n")
                    else:
                        f.write("max epoch tercapai.\n")

                    f.write(f"\nBobot akhir : {self.w_[1:]}\n")
                    f.write(f"Bias akhir : {self.w_[0]}\n")
                    break
