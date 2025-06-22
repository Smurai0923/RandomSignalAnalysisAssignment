#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include "qcustomplot.h"
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <random>

const double PI = 3.14159265358979323846;


// 生成信号并添加高斯白噪声
void computeSignal(std::vector<double>& x, int N, double f1, double f2, double noiseStdDev) {
    x.resize(N);
    // 随机数生成器，用于生成高斯白噪声
    std::random_device rd;
    std::mt19937 gen(rd());  // 随机数引擎
    std::normal_distribution<> d(0.0, noiseStdDev);  // 高斯分布，均值0，标准差为 noiseStdDev
    // 生成信号并添加噪声
    for (int n = 0; n < N; ++n) {
        double signal = 10 * sin(2 * PI * f1 * n + PI / 3) + 4 * sin(2 * PI * f2 * n + PI / 4);
        double noise = d(gen);  // 生成高斯噪声
        x[n] = signal + noise;  // 将噪声添加到信号中
    }
}

// 周期图法（FFT）
void computePeriodogram(const std::vector<double>& x, std::vector<double>& psd) {
    int N = x.size();
    psd.clear();
    psd.resize(N);
    for (int k = 0; k < N; ++k) {
        std::complex<double> sum(0, 0);
        for (int n = 0; n < N; ++n)
            sum += std::polar(x[n], -2 * PI * k * n / N);
        psd[k] = std::norm(sum) / N;
    }
}

//计算自相关函数
void computeAutocorrelation(const std::vector<double>& x, std::vector<double>& r, int p) {
    int N = x.size();
    r.resize(p + 1);
    for (int k = 0; k <= p; ++k) {
        double sum = 0.0;
        for (int n = 0; n < N - k; ++n) {
            sum += x[n] * x[n + k];
        }
        r[k] = sum / (N - k); // 修正归一化因子
    }
}

// 使用自相关法计算功率谱密度
void computePSD_ACF(const std::vector<double>& r, std::vector<double>& psd, int fft_len) {
    psd.resize(fft_len);
    for (int i = 0; i < fft_len; ++i) {
        double freq = i * 1.0 / fft_len;
        std::complex<double> R(0, 0);
        for (int k = 0; k < r.size(); ++k)
            R += std::polar(r[k], -2 * PI * freq * k);
        psd[i] = std::real(R); // 取实部
    }
}

// AR-Yule-Walker（高斯消元）
void solveYuleWalker(const std::vector<double>& r, std::vector<double>& a, double& noiseVar) {
    int p = r.size() - 1;
    std::vector<std::vector<double>> R(p, std::vector<double>(p));
    std::vector<double> rhs(p);
    for (int i = 0; i < p; ++i) {
        rhs[i] = -r[i + 1];
        for (int j = 0; j < p; ++j)
            R[i][j] = r[std::abs(i - j)];
    }

    for (int i = 0; i < p; ++i) {
        double pivot = R[i][i];
        for (int j = i; j < p; ++j) R[i][j] /= pivot;
        rhs[i] /= pivot;
        for (int k = i + 1; k < p; ++k) {
            double factor = R[k][i];
            for (int j = i; j < p; ++j) R[k][j] -= factor * R[i][j];
            rhs[k] -= factor * rhs[i];
        }
    }

    a.resize(p + 1);
    a[0] = 1.0;
    for (int i = p - 1; i >= 0; --i) {
        a[i + 1] = rhs[i];
        for (int j = i + 1; j < p; ++j)
            a[i + 1] -= R[i][j] * a[j + 1];
    }

    noiseVar = r[0];
    for (int i = 1; i < r.size(); ++i)
        noiseVar += a[i] * r[i];
}

// AR-LD（Levinson-Durbin）
void levinsonDurbin(const std::vector<double>& r, std::vector<double>& a, double& noiseVar) {
    int p = r.size() - 1;
    a.assign(p + 1, 0.0);
    std::vector<double> a_prev(p + 1, 0.0);
    a[0] = 1.0;
    noiseVar = r[0];

    for (int m = 1; m <= p; ++m) {
        double k = 0;
        for (int i = 1; i < m; ++i)
            k += a[i] * r[m - i];
        k = (r[m] + k) / noiseVar;

        for (int i = 1; i < m; ++i)
            a[i] = a[i] - k * a_prev[m - i];
        a[m] = -k;
        a_prev = a;
        noiseVar *= (1.0 - k * k);
    }
}

// AR-Burg
void burgAlgorithm(const std::vector<double>& x, int order, std::vector<double>& a, double& noiseVar) {
    int N = x.size();
    if (order >= N) order = N - 1;

    std::vector<double> f(x);        // forward prediction error
    std::vector<double> b(x);        // backward prediction error

    a.assign(order + 1, 0.0);
    a[0] = 1.0;

    std::vector<double> a_prev = a;
    noiseVar = std::inner_product(x.begin(), x.end(), x.begin(), 0.0) / N;

    for (int m = 1; m <= order; ++m) {
        double num = 0.0, den = 0.0;
        for (int n = m; n < N; ++n) {
            num += f[n] * b[n - 1];
            den += f[n] * f[n] + b[n - 1] * b[n - 1];
        }

        if (den <= 0.0) break;

        double k = -2.0 * num / den;

        // 更新 AR 系数（注意方向）
        for (int i = 1; i < m; ++i) {
            a[i] = a_prev[i] + k * a_prev[m - i];
        }
        a[m] = k;

        // 更新误差序列
        for (int n = N - 1; n >= m; --n) {
            double temp_f = f[n] + k * b[n - 1];
            double temp_b = b[n - 1] + k * f[n];
            f[n] = temp_f;
            b[n - 1] = temp_b;
        }

        a_prev = a;
        noiseVar *= (1.0 - k * k);
    }

    // 最后统一反号以匹配 PSD 中 1 / |1 + a1 z^-1 + ...|² 模型
    for (int i = 1; i <= order; ++i)
        a[i] = -a[i];
}

// 根据 AR 系数计算 PSD
void computePSD_AR(const std::vector<double>& a, double noiseVar, std::vector<double>& psd, int fft_len) {
    psd.resize(fft_len);
    for (int i = 0; i < fft_len; ++i) {
        double freq = i * 1.0 / fft_len;
        std::complex<double> denom(1.0, 0.0);
        for (size_t k = 1; k < a.size(); ++k)
            denom += std::polar(a[k], -2 * PI * freq * k);
        psd[i] = noiseVar / std::norm(denom);
    }
    double maxval = *std::max_element(psd.begin(), psd.end());
    for (auto& v : psd) v /= maxval;
}

// 绘图
void plotPSD(QCustomPlot* plot, const std::vector<double>& psd) {
    QVector<double> freq, power;
    int N = psd.size();

    std::vector<int> peakIndices;
    for (int i = 1; i < N - 1; ++i) {
        if (psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > 0) {
            peakIndices.push_back(i);
        }
    }

    for (int i = 0; i < N / 2; ++i) {
        freq.push_back(i * 0.5 / (N / 2)); // Normalized frequency in [0, 0.5]
        power.push_back(psd[i]);
    }

    plot->clearGraphs();
    plot->clearItems();
    plot->addGraph();
    plot->graph(0)->setData(freq, power);
    plot->xAxis->setLabel("Normalized Frequency");
    plot->yAxis->setLabel("Normalized Power");
    plot->rescaleAxes();

    // 筛选出频率不为 0 的峰值
    std::vector<std::pair<int, double>> peaksWithPower;
    for (int peakIndex : peakIndices) {
        if (freq[peakIndex] > 0 && power[peakIndex] > 0)
            peaksWithPower.emplace_back(peakIndex, power[peakIndex]);
    }

    std::sort(peaksWithPower.begin(), peaksWithPower.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    for (int i = 0; i < std::min(2, (int)peaksWithPower.size()); ++i) {
        int peakIndex = peaksWithPower[i].first;
        double peakFreq = freq[peakIndex];
        double peakPower = power[peakIndex];

        auto *textLabel = new QCPItemText(plot);
        textLabel->position->setCoords(peakFreq, peakPower);
        textLabel->setText(QString("Peak: %1").arg(peakFreq, 0, 'f', 3));
        textLabel->setFont(QFont("Arial", 10));
        textLabel->setColor(Qt::red);
        textLabel->setPositionAlignment(Qt::AlignTop | Qt::AlignHCenter);

        auto *arrow = new QCPItemLine(plot);
        arrow->start->setCoords(peakFreq, peakPower);
        arrow->end->setCoords(peakFreq, peakPower * 0.9);
        arrow->setHead(QCPLineEnding::esSpikeArrow);
    }

    plot->replot();
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QMainWindow window;

    // 中央控件和布局
    auto* centralWidget = new QWidget;
    auto* layout = new QVBoxLayout(centralWidget);
    window.setCentralWidget(centralWidget);

    // 下拉框选择方法
    auto* comboBox = new QComboBox;
    comboBox->addItems({"Periodogram", "Autocorrelation", "AR-YuleWalker", "AR-LD", "AR-Burg"});

    // f1, f2, order 控件
    auto* f1Input = new QDoubleSpinBox;
    f1Input->setRange(0.0, 0.5);
    f1Input->setSingleStep(0.01);
    f1Input->setValue(0.1);

    auto* f2Input = new QDoubleSpinBox;
    f2Input->setRange(0.0, 0.5);
    f2Input->setSingleStep(0.01);
    f2Input->setValue(0.3);

    auto* orderInput = new QSpinBox;
    orderInput->setRange(1, 100);
    orderInput->setValue(20);

    // 添加控件到布局
    layout->addWidget(comboBox);
    layout->addWidget(f1Input);
    layout->addWidget(f2Input);
    layout->addWidget(orderInput);

    // 图形绘制控件
    auto* plot = new QCustomPlot;
    plot->addGraph();
    layout->addWidget(plot);

    // 信号处理变量
    int N = 256;
    std::vector<double> x, psd, r, a;
    double noiseVar = 0.0;
    double noiseStdDev = 2.0;

    // 更新图形函数
    auto updatePlot = [&](int index) {
        double f1 = f1Input->value();
        double f2 = f2Input->value();
        int order = orderInput->value();

        computeSignal(x, N, f1, f2, noiseStdDev);
        computeAutocorrelation(x, r, order);

        if (index == 0)
            computePeriodogram(x, psd);
        else if (index == 1)
            computePSD_ACF(r, psd, N);
        else if (index == 2) {
            solveYuleWalker(r, a, noiseVar);
            computePSD_AR(a, noiseVar, psd, N);
        } else if (index == 3) {
            levinsonDurbin(r, a, noiseVar);
            computePSD_AR(a, noiseVar, psd, N);
        } else if (index == 4) {
            burgAlgorithm(x, order, a, noiseVar);
            computePSD_AR(a, noiseVar, psd, N);
        }

        plotPSD(plot, psd);
    };

    // 信号连接
    QObject::connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), updatePlot);
    QObject::connect(f1Input, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double){ updatePlot(comboBox->currentIndex()); });
    QObject::connect(f2Input, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double){ updatePlot(comboBox->currentIndex()); });
    QObject::connect(orderInput, QOverload<int>::of(&QSpinBox::valueChanged), [&](int){ updatePlot(comboBox->currentIndex()); });

    updatePlot(0);  // 默认显示

    window.resize(800, 600);
    window.show();

    return app.exec();
}
