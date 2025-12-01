{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMxf4itl3/450z/JTbkaDyY",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/junsang4692-cyber/janus-sign-ai-simulation/blob/main/%ED%91%9C%EC%A7%80%ED%8C%90app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 1] 라이브러리 임포트 & 기본 설정"
      ],
      "metadata": {
        "id": "e0gYGelwT9Lq"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yrZtjf_ISXvp",
        "outputId": "53a5dfc2-6eca-4342-d6b1-94bc2b3e68a4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "TensorFlow: 2.19.0\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import classification_report\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers\n",
        "\n",
        "np.random.seed(42)\n",
        "tf.random.set_seed(42)\n",
        "\n",
        "print(\"TensorFlow:\", tf.__version__)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 2] 전역 파라미터 & 기본 유틸 (비트 ↔ 3×3 그리드)"
      ],
      "metadata": {
        "id": "UT9Ky63PUCVg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 이미지/패턴 설정\n",
        "IMG_SIZE = 32\n",
        "GRID_SIZE = 3\n",
        "CELL_SIZE = IMG_SIZE // GRID_SIZE  # 대략 10~11px\n",
        "\n",
        "# 사용할 패턴 클래스 수 (테스트용으로 32개. 이론적 최대는 512)\n",
        "N_CLASSES = 32\n",
        "\n",
        "def id_to_pattern_bits(pattern_id, num_bits=9):\n",
        "    \"\"\"정수 ID(0~511) -> 9비트 배열\"\"\"\n",
        "    bits = [int(x) for x in np.binary_repr(pattern_id, width=num_bits)]\n",
        "    return np.array(bits, dtype=np.int32)\n",
        "\n",
        "def bits_to_grid(bits):\n",
        "    \"\"\"길이 9 비트 -> 3x3 배열\"\"\"\n",
        "    return bits.reshape((GRID_SIZE, GRID_SIZE))\n"
      ],
      "metadata": {
        "id": "6JtCQlFUSZCF"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 3] 야누스 표지판 NIR 반사 패턴 → 이미지로 합성"
      ],
      "metadata": {
        "id": "yUq0PA6sUJYN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def generate_pattern_image(\n",
        "    pattern_id,\n",
        "    img_size=IMG_SIZE,\n",
        "    cell_size=CELL_SIZE,\n",
        "    high_intensity=0.9,  # NIR 고반사 셀(1)의 평균 밝기\n",
        "    low_intensity=0.2,   # NIR 저반사 셀(0)의 평균 밝기\n",
        "    noise_level=0.08,\n",
        "    blur=False\n",
        "):\n",
        "    \"\"\"\n",
        "    주어진 3x3 비트 패턴을 근적외선(NIR) intensity map으로 합성.\n",
        "    - 1: high_intensity, 0: low_intensity\n",
        "    - 전체에 가우시안 노이즈 추가 (라이다 노이즈 모사)\n",
        "    \"\"\"\n",
        "    bits = id_to_pattern_bits(pattern_id)\n",
        "    grid = bits_to_grid(bits)\n",
        "\n",
        "    img = np.zeros((img_size, img_size), dtype=np.float32)\n",
        "\n",
        "    for r in range(GRID_SIZE):\n",
        "        for c in range(GRID_SIZE):\n",
        "            val = high_intensity if grid[r, c] == 1 else low_intensity\n",
        "            y_start = r * cell_size\n",
        "            y_end   = (r + 1) * cell_size\n",
        "            x_start = c * cell_size\n",
        "            x_end   = (c + 1) * cell_size\n",
        "            img[y_start:y_end, x_start:x_end] = val\n",
        "\n",
        "    # NIR 신호의 불안정성/노이즈\n",
        "    noise = np.random.normal(0, noise_level, size=img.shape).astype(np.float32)\n",
        "    img = np.clip(img + noise, 0.0, 1.0)\n",
        "\n",
        "    if blur:\n",
        "        from scipy.ndimage import gaussian_filter\n",
        "        img = gaussian_filter(img, sigma=0.5)\n",
        "\n",
        "    return img\n"
      ],
      "metadata": {
        "id": "8sxmZLsrSZFB"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 4] 학습용 데이터셋 생성 (깨끗한/노이즈 포함 패턴 이미지)"
      ],
      "metadata": {
        "id": "GABSD2R7UmVi"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def create_dataset(\n",
        "    n_classes=N_CLASSES,\n",
        "    samples_per_class=300,\n",
        "    noise_level=0.08\n",
        "):\n",
        "    X, y = [], []\n",
        "\n",
        "    for pattern_id in range(n_classes):\n",
        "        for _ in range(samples_per_class):\n",
        "            img = generate_pattern_image(pattern_id, noise_level=noise_level)\n",
        "            X.append(img)\n",
        "            y.append(pattern_id)\n",
        "\n",
        "    X = np.array(X, dtype=np.float32)\n",
        "    y = np.array(y, dtype=np.int64)\n",
        "\n",
        "    # CNN 입력을 위한 채널 차원 추가\n",
        "    X = X[..., np.newaxis]  # (N, H, W, 1)\n",
        "\n",
        "    return X, y\n",
        "\n",
        "\n",
        "X, y = create_dataset()\n",
        "print(\"X:\", X.shape, \"y:\", y.shape)\n",
        "\n",
        "# train / val / test 분할 (70 / 15 / 15)\n",
        "X_temp, X_test, y_temp, y_test = train_test_split(\n",
        "    X, y, test_size=0.15, random_state=42, stratify=y\n",
        ")\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp\n",
        ")\n",
        "\n",
        "print(\"Train:\", X_train.shape, \"Val:\", X_val.shape, \"Test:\", X_test.shape)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xKcdQNuBSZI9",
        "outputId": "c405dd38-2919-42af-9713-ce92d62fb785"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "X: (9600, 32, 32, 1) y: (9600,)\n",
            "Train: (6719, 32, 32, 1) Val: (1441, 32, 32, 1) Test: (1440, 32, 32, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 5] 샘플 이미지 확인 (시각 검증용)"
      ],
      "metadata": {
        "id": "Q9wjwxwwUpd2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def show_samples(num_samples=6):\n",
        "    idx = np.random.choice(len(X_train), num_samples, replace=False)\n",
        "    plt.figure(figsize=(10, 2))\n",
        "    for i, k in enumerate(idx):\n",
        "        img = X_train[k, ..., 0]\n",
        "        label = y_train[k]\n",
        "        plt.subplot(1, num_samples, i+1)\n",
        "        plt.imshow(img, cmap=\"gray\")\n",
        "        plt.title(f\"ID {label}\")\n",
        "        plt.axis(\"off\")\n",
        "    plt.suptitle(\"Train samples (NIR 패턴 이미지)\")\n",
        "    plt.show()\n",
        "\n",
        "show_samples()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 371
        },
        "id": "8uM1__1KSZLm",
        "outputId": "aeff7c04-5265-4bb5-c3a7-f811e64c9ad3"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54056 (\\N{HANGUL SYLLABLE PAE}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 53556 (\\N{HANGUL SYLLABLE TEON}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51060 (\\N{HANGUL SYLLABLE I}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 48120 (\\N{HANGUL SYLLABLE MI}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51648 (\\N{HANGUL SYLLABLE JI}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x200 with 6 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAxsAAACsCAYAAAAT+bGZAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVclJREFUeJztvXm0VdWVvj2viiBeQWmUqEEQUFFQI4iAdKKAIIpdJEYSxVhFRqJkDC2jMcRSo2VMKiVWjKVoRSJComIMCipdQBGlt4lEhWBvRAMKYhkIzfn+yI/9PeuFuz1c7r5XzfuM4Rjrss7ZzVpzzbWP851zV5RKpVIYY4wxxhhjTA2zS11fgDHGGGOMMeaLiX9sGGOMMcYYYwrBPzaMMcYYY4wxheAfG8YYY4wxxphC8I8NY4wxxhhjTCH4x4YxxhhjjDGmEPxjwxhjjDHGGFMI/rFhjDHGGGOMKQT/2DDGGGOMMcYUgn9sGGM+N1xwwQXRqlWrur6MOqeuxmHLli3RoUOHuOGGG2r93P/MbNy4Mb785S/HbbfdVteXYowxO4x/bBhjdpqKioqy/ps9e3ZdX6rZCX7zm9/EW2+9FRdffHH2b2PHjo2Kiopo0KBBvPPOO9t8p0+fPtGhQ4fk31q1ahWDBw9O/k1tpVGjRtG7d++YMmVKMTfzOaJevXpx6aWXxg033BDr16+v68sxxpgdwj82jDE7zbhx45L/+vXrt91/b9++/U6d584774xXXnmlJi7ZVIOf/exn8bWvfS0aN268Td+GDRviJz/5yU4dv1+/fjFu3Li455574vvf/378+c9/jlNPPTWmTp36qd9t0aJFVFZWbve/Bg0axK9+9asd+tz2GDp0aDRs2HC7323YsGGcf/75hXwuImL48OGxatWqmDBhws4MsTHG1Dq71fUFGGM+/wwbNiz5e968eTF9+vRt/l355JNPomHDhmWfp169etW6PrPzPPvss/H888/Hz3/+8+32H3300XHnnXfGD37wg9h///2rdY5DDjkksZmzzjorDj/88LjllltiwIABud/dtGlTrFmzJnbbbdtt7corr4wtW7bs0Oe2x+bNm+Phhx+Ok046aZu+xx9/PO69995CPhcRsffee0f//v1j7NixceGFF1Z5jcYY81nDkQ1jTK2wVU6zePHi6NWrVzRs2DCuuuqqiIiYNGlSnHLKKbH//vtH/fr1o02bNvHjH/84Nm/enBxDcxVef/31qKioiP/8z/+MMWPGRJs2baJ+/fpx7LHHxsKFCz/1mjZu3BjXXntttGvXLho0aBBNmzaNHj16xPTp07PPvPDCC3HBBRfEwQcfHA0aNIgWLVrEhRdeGKtXr06Odc0110RFRUUsW7Yshg0bFo0bN47mzZvHj370oyiVSvHWW2/FkCFDolGjRtGiRYttHtpnz54dFRUVcd9998VVV10VLVq0iD333DNOO+20eOuttz71XrZs2RKjR4+OI444Iho0aBD77bdfjBgxIj788MPkc4sWLYoBAwZEs2bNYo899ojWrVuX9fD6+9//Pnbffffo1avXdvuvuuqq2Lx5805HN0j79u2jWbNmsWLFiho75ueZfv36xVNPPRUffPBBXV+KMcaUjSMbxphaY/Xq1TFw4MD42te+FsOGDYv99tsvIv6h+6+srIxLL700Kisr4w9/+ENcffXV8dFHH8XPfvazTz3uhAkTYt26dTFixIioqKiIn/70p3HmmWfGq6++mhsNueaaa+LGG2+Miy66KLp06RIfffRRLFq0KJYsWZJJwaZPnx6vvvpqDB8+PFq0aBFLly6NMWPGxNKlS2PevHlRUVGRHHPo0KHRvn37+MlPfhJTpkyJ66+/Ppo0aRJ33HFH9O3bN2666aYYP358/Nu//Vsce+yx2zy833DDDVFRURFXXHFFvP/++zF69Og46aST4rnnnos99tijynsZMWJEjB07NoYPHx4jR46M1157LW699dZ49tlnY+7cuVGvXr14//33o3///tG8efO48sorY++9947XX389fve7333qGD/99NPRoUOHKsezdevW8c1vfjPuvPPOuPLKK6sd3SBr166NDz/8MNq0abPTx/oi0KlTpyiVSvH0009vk/NijDGfVfxjwxhTa6xcuTJuv/32GDFiRPLvEyZMSB6kv/3tb8e3v/3tuO222+L666+P+vXr5x73zTffjOXLl8c+++wTERGHHnpoDBkyJKZOnZr7UDZlypQYNGhQjBkzpsrPfOc734nLLrss+beuXbvGueeeG0899VT07Nkz6evSpUvccccdERHxr//6r9GqVau47LLL4sYbb4wrrrgiIiLOPffc2H///eNXv/rVNj82Pvjgg3jppZdir732ioiIY445Js4555y48847Y+TIkdu9xqeeeiruuuuuGD9+fHz961/P/v2EE06Ik08+OR544IH4+te/Hk8//XR8+OGHMW3atOjcuXP2ueuvv77K+9/Kyy+/HMcdd1zuZ374wx/GPffcEzfddFPccsstn3pMZf369bFq1aoolUrx5ptvxqhRo2Lz5s1x9tln7/CxvogcfPDBERHxpz/9yT82jDGfGyyjMsbUGvXr14/hw4dv8+/8obFu3bpYtWpV9OzZMz755JN4+eWXP/W4Q4cOzX5oRET2A+DVV1/N/d7ee+8dS5cujeXLl1f5GV7b1ofhrl27RkTEkiVLtvn8RRddlLV33XXX6Ny5c5RKpfjWt76VnPfQQw/d7vV985vfzH5oREScffbZ8aUvfSkeffTRKq/xgQceiMaNG0e/fv1i1apV2X+dOnWKysrKmDVrVnbeiIjJkyfHxo0bqzze9li9enUyxtvj4IMPjm984xsxZsyYePfdd3fo+BER//u//xvNmzePfffdNzp37hwzZ86M73//+3HppZfu8LG+iGwd/1WrVtXxlRhjTPn4x4YxptY44IADYvfdd9/m35cuXRpnnHFGNG7cOBo1ahTNmzfPEoXXrl37qcdt2bJl8vfWhzLNV1Cuu+66WLNmTRxyyCHRsWPHuPzyy+OFF15IPvPBBx/E9773vdhvv/1ijz32iObNm0fr1q2rvDa9lsaNG0eDBg2iWbNm2/z79q6vXbt2yd8VFRXRtm3beP3116u8j+XLl8fatWtj3333jebNmyf/ffzxx/H+++9HRETv3r3jrLPOimuvvTaaNWsWQ4YMibvvvjs2bNhQ9SCBUqn0qZ8ZNWpUbNq0qVq5G0OGDInp06fHlClTshyYTz75JHbZxVtVxP8//irdM8aYzzKWURljao3t5RysWbMmevfuHY0aNYrrrrsu2rRpEw0aNIglS5bEFVdckVsdaCu77rrrdv/90x6Oe/XqFStWrIhJkybFtGnT4q677oqbb745br/99ixCcc4558TTTz8dl19+eRx99NFRWVkZW7ZsiZNPPnm717a9a6nu9ZXLli1bYt99943x48dvt7958+YR8Y+H1IkTJ8a8efPikUceialTp8aFF14YP//5z2PevHlRWVlZ5TmaNm36qT/eIv4R3Rg2bFiMGTMmrrzyyh26jwMPPDCrzDRo0KBo1qxZXHzxxXHCCSfEmWeeuUPH+iKydfz1h6sxxnyW8Y8NY0ydMnv27Fi9enX87ne/S/IXXnvttVo5f5MmTWL48OExfPjw+Pjjj6NXr15xzTXXxEUXXRQffvhhzJw5M6699tq4+uqrs+/kya52Fj12qVSKP//5z3HkkUdW+Z02bdrEjBkz4vjjj89NIt9K165do2vXrnHDDTfEhAkT4rzzzovf/va3iQRMOeyww8qek1GjRsW9994bN910U1mfr4oRI0bEzTffHKNGjYozzjjjn/7/6G8d/519X40xxtQmjk0bY+qUrf/Xn/+X/+9//3vcdttthZ9by9dWVlZG27ZtM1nR9q4tImL06NGFXdM999wT69aty/6eOHFivPvuuzFw4MAqv3POOefE5s2b48c//vE2fVvfKxHxj/8zrvdy9NFHR0R8qpSqW7du8eKLL5YluWrTpk0MGzYs7rjjjli5cuWnfr4qdtttt7jsssvipZdeikmTJlX7OF8UFi9eHBUVFdGtW7e6vhRjjCkbRzaMMXVK9+7dY5999onzzz8/Ro4cGRUVFTFu3Lgakxjlcfjhh0efPn2iU6dO0aRJk1i0aFFMnDgxLr744oiIaNSoUfTq1St++tOfxsaNG+OAAw6IadOmFRp1adKkSfTo0SOGDx8e7733XowePTratm0b//Iv/1Lld3r37h0jRoyIG2+8MZ577rno379/1KtXL5YvXx4PPPBA3HLLLXH22WfHr3/967jtttvijDPOiDZt2sS6devizjvvjEaNGsWgQYNyr2vIkCHx4x//OJ544ono37//p97HD3/4wxg3bly88sorccQRR+zwOGzlggsuiKuvvjpuuummOP3006t9nC8C06dPj+OPPz6aNm1a15dijDFl4x8bxpg6pWnTpjF58uS47LLLYtSoUbHPPvvEsGHD4sQTT/zUt0bvLCNHjoyHH344pk2bFhs2bIiDDjoorr/++rj88suzz0yYMCEuueSS+OUvfxmlUin69+8fjz32WI28R2J7XHXVVfHCCy/EjTfeGOvWrYsTTzwxbrvttk990/rtt98enTp1ijvuuCOuuuqq2G233aJVq1YxbNiwOP744yPiHz9KFixYEL/97W/jvffei8aNG0eXLl1i/PjxWdJ7VXTq1CmOPPLIuP/++8v6sdG2bdsYNmxY/PrXvy7/5rfDHnvsERdffHFcc801MXv27OjTp89OHe/zytq1a2PatGm1EvEzxpgapWSMMabOmTVrVikiSg888EBdX0qV3HPPPaW99tqr9OGHH9b1pWxD06ZNSxs3btxu3xVXXFG68847d+hz2+Oss84qTZ8+fbt9jz32WOm8884r5HOlUql08803l770pS+VPvnkkyqvzxhjPos4Z8MYY0xZnHfeedGyZcv45S9/WdeX8k/Fxo0b47/+679i1KhRZRUAMMaYzxKWURljjCmLXXbZJV588cW6vowqqaok7Pr16+PWW2/d4c9tj9NPPz12223brXPTpk1JTklNfq5evXrx5ptv5l6XMcZ8VvGPDWOMMZ97yn2r9s68fXvixIl18jljjPk8U1Eq1ULJF2OMMcYYY8w/Hc7ZMMYYY4wxxhTC5+7HxtixY6OioiIWLVqU/ds111wTFRUV2X8NGzaMli1bxqmnnhp33313WS+hioj4+OOP49///d/j5JNPjiZNmkRFRUWMHTt2u5/l+fS/fv361cStmjIp0iYWLlwYF198cRxxxBGx5557RsuWLeOcc86JZcuWbfPZBQsWxHe+853o1KlT1KtX75/+bcd1yWfFT0RE3H///dG1a9fYe++9o2nTptG7d++YMmXKzt6iqQGKtJMLLrggd5945513irotsxN8VvYT89mhSJtQbrjhhqioqIgOHTrU1OV/JvhC5Wz8z//8T1RWVsaGDRvinXfeialTp8aFF14Yo0ePjsmTJ8eXv/zl3O+vWrUqrrvuumjZsmUcddRRMXv27Co/O27cuG3+bdGiRXHLLbeUVYPe1A47axM33XRTzJ07N7761a/GkUceGStXroxbb701jjnmmJg3b17iEB599NG466674sgjj4yDDz7YG8hnlNr0E7/4xS9i5MiRccopp8RPfvKTWL9+fYwdOzYGDx4cDz74YJx55pk1fHempthZOxkxYkScdNJJyb+VSqX49re/Ha1atYoDDjigyMs3BVCb+4n5fLCzNkHefvvt+I//+I/Yc889C7ziOqKOS+/uMHfffXcpIkoLFy7M/u3f//3fSxFR+utf/7rN5++9997SLrvsUjruuOM+9djr168vvfvuu6VSqVRauHBhKSJKd999d9nX9q1vfatUUVFReuutt8r+jtl5irSJuXPnljZs2JD827Jly0r169dPauCXSqXSypUrsxr43/3ud0ufw+X1heGz4ifa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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 6] CNN 모델 정의 & 학습 (AI 인식부)"
      ],
      "metadata": {
        "id": "WPxG_XlHUtxQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "input_shape = (IMG_SIZE, IMG_SIZE, 1)\n",
        "\n",
        "model = keras.Sequential([\n",
        "    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),\n",
        "    layers.MaxPooling2D((2,2)),\n",
        "\n",
        "    layers.Conv2D(64, (3,3), activation='relu', padding='same'),\n",
        "    layers.MaxPooling2D((2,2)),\n",
        "\n",
        "    layers.Conv2D(128, (3,3), activation='relu', padding='same'),\n",
        "    layers.MaxPooling2D((2,2)),\n",
        "\n",
        "    layers.Flatten(),\n",
        "    layers.Dense(128, activation='relu'),\n",
        "    layers.Dropout(0.3),\n",
        "    layers.Dense(N_CLASSES, activation='softmax')\n",
        "])\n",
        "\n",
        "model.compile(\n",
        "    optimizer=keras.optimizers.Adam(1e-3),\n",
        "    loss='sparse_categorical_crossentropy',\n",
        "    metrics=['accuracy']\n",
        ")\n",
        "\n",
        "model.summary()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 505
        },
        "id": "FxKWXJSfSZOc",
        "outputId": "7eee602b-14e5-494a-be9b-cdd7e53b261d"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │           \u001b[38;5;34m320\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m16\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m18,496\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m64\u001b[0m)       │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │        \u001b[38;5;34m73,856\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2048\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)            │       \u001b[38;5;34m262,272\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)            │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m)             │         \u001b[38;5;34m4,128\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">320</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)       │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2048</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)            │       <span style=\"color: #00af00; text-decoration-color: #00af00\">262,272</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)             │         <span style=\"color: #00af00; text-decoration-color: #00af00\">4,128</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m359,072\u001b[0m (1.37 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">359,072</span> (1.37 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m359,072\u001b[0m (1.37 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">359,072</span> (1.37 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "L0_p737UUvUa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "EPOCHS = 12\n",
        "BATCH_SIZE = 64\n",
        "\n",
        "history = model.fit(\n",
        "    X_train, y_train,\n",
        "    validation_data=(X_val, y_val),\n",
        "    epochs=EPOCHS,\n",
        "    batch_size=BATCH_SIZE\n",
        ")\n",
        "\n",
        "plt.figure(figsize=(12,4))\n",
        "plt.subplot(1,2,1)\n",
        "plt.plot(history.history['loss'], label='train')\n",
        "plt.plot(history.history['val_loss'], label='val')\n",
        "plt.title(\"Loss\")\n",
        "plt.legend()\n",
        "\n",
        "plt.subplot(1,2,2)\n",
        "plt.plot(history.history['accuracy'], label='train')\n",
        "plt.plot(history.history['val_accuracy'], label='val')\n",
        "plt.title(\"Accuracy\")\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n",
        "test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)\n",
        "print(f\"깨끗한 테스트 세트 정확도: {test_acc*100:.2f}%\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 847
        },
        "id": "qB9-jru-SZRC",
        "outputId": "53a58eaf-5a90-4811-87a6-c107d9750cec"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 150ms/step - accuracy: 0.3957 - loss: 2.1337 - val_accuracy: 1.0000 - val_loss: 0.0012\n",
            "Epoch 2/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 164ms/step - accuracy: 0.9808 - loss: 0.0641 - val_accuracy: 1.0000 - val_loss: 2.0209e-04\n",
            "Epoch 3/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 142ms/step - accuracy: 0.9874 - loss: 0.0383 - val_accuracy: 1.0000 - val_loss: 1.0315e-04\n",
            "Epoch 4/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 145ms/step - accuracy: 0.9918 - loss: 0.0262 - val_accuracy: 1.0000 - val_loss: 2.6905e-05\n",
            "Epoch 5/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 152ms/step - accuracy: 0.9929 - loss: 0.0215 - val_accuracy: 1.0000 - val_loss: 2.6107e-05\n",
            "Epoch 6/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m20s\u001b[0m 151ms/step - accuracy: 0.9950 - loss: 0.0122 - val_accuracy: 1.0000 - val_loss: 3.9874e-07\n",
            "Epoch 7/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m16s\u001b[0m 151ms/step - accuracy: 0.9973 - loss: 0.0067 - val_accuracy: 1.0000 - val_loss: 1.0560e-05\n",
            "Epoch 8/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 141ms/step - accuracy: 0.9953 - loss: 0.0124 - val_accuracy: 1.0000 - val_loss: 6.6884e-07\n",
            "Epoch 9/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 148ms/step - accuracy: 0.9957 - loss: 0.0134 - val_accuracy: 1.0000 - val_loss: 9.4944e-07\n",
            "Epoch 10/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 140ms/step - accuracy: 0.9973 - loss: 0.0068 - val_accuracy: 1.0000 - val_loss: 2.0875e-06\n",
            "Epoch 11/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 145ms/step - accuracy: 0.9982 - loss: 0.0044 - val_accuracy: 1.0000 - val_loss: 3.9957e-07\n",
            "Epoch 12/12\n",
            "\u001b[1m105/105\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 151ms/step - accuracy: 0.9984 - loss: 0.0079 - val_accuracy: 1.0000 - val_loss: 7.3738e-06\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x400 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA9UAAAF2CAYAAABgXbt2AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAY4RJREFUeJzt3Xt8VOWB//HvmcncQsiNhIRLaFARpSooCItiq2tqBJtVWy0ilouV/nTJrpilFpSLSjXVKkUtSr0g2sKKVXTZxWJpFFsqioK29QIKiglIAuGSeybJzPn9kcwkQyYh95lkPu/Xa17JnHnOmWemliff89wM0zRNAQAAAACAdrOEugIAAAAAAPRWhGoAAAAAADqIUA0AAAAAQAcRqgEAAAAA6CBCNQAAAAAAHUSoBgAAAACggwjVAAAAAAB0EKEaAAAAAIAOIlQDAAAAANBBhGoAAAAAADqIUA30ImvWrJFhGPrggw9CXRUAABDEE088IcMwNGHChFBXBUAPIVQDAAAAXWTt2rVKT0/Xjh07tHfv3lBXB0APIFQDAAAAXeCrr77SO++8o+XLlys5OVlr164NdZWCqqioCHUVgD6FUA30MR9++KEmT56s2NhYxcTE6PLLL9e7774bUKa2tlb33nuvRowYIafTqQEDBmjSpEnasmWLv0xhYaFmz56toUOHyuFwaNCgQbr66qu1f//+Hv5EAAD0DmvXrlVCQoKuuuoqXXfddUFD9YkTJ3THHXcoPT1dDodDQ4cO1YwZM1RcXOwvU11drXvuuUdnnnmmnE6nBg0apB/84Afat2+fJGnr1q0yDENbt24NuPb+/ftlGIbWrFnjPzZr1izFxMRo3759mjJlivr376/p06dLkv7617/q+uuv17Bhw+RwOJSWlqY77rhDVVVVzeq9e/du/ehHP1JycrJcLpdGjhypu+++W5L01ltvyTAMvfrqq83OW7dunQzD0Pbt29v9fQK9RVSoKwCg63zyySe65JJLFBsbqzvvvFM2m02//e1vdemll+rtt9/2z++65557lJubq1tuuUXjx49XaWmpPvjgA+3atUvf+973JEk//OEP9cknn+g//uM/lJ6ersOHD2vLli3Kz89Xenp6CD8lAADhae3atfrBD34gu92uadOm6cknn9T777+vCy+8UJJUXl6uSy65RJ999pluvvlmXXDBBSou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uPfee/WjH/1I//jHPzRlyhRNnz5dx44d6/T32xqGf/cw33yuA8er5PWaslhYBh8AgO7im0+dFGNXPwd/9gDou6pqPRq15I2QvPen92Uq2n7qf2Pj4uJkt9sVHR2t1NRUSdLu3bslSffdd5++973v+csmJiZq9OjR/ufLli3Tq6++qo0bNyo7O7vF95g1a5amTZsmSXrggQf02GOPaceOHbryyis79Nnagp7qHjYozimrxVCNx6uisupQVwcAgD6NRcoAoHcYN25cwPPy8nLNnz9fZ599tuLj4xUTE6PPPvvslD3V5513nv/3fv36KTY2VocPH+6WOvtwy7aHRVktGhLvUv6xSuUfrdSgOIaiAQDQXfzbaSUQqgH0bS6bVZ/elxmy9+6sk1fxnj9/vrZs2aKHH35YZ5xxhlwul6677jrV1NS0eh2bzRbw3DAMeb3eTtevNYTqEEhLrA/VBcerNCHUlQEAoA8raFgYlEXKAPR1hmG0aQh2qNntdnk8nlOW+9vf/qZZs2bp2muvlVTfc71///5url3HMPw7BBpXAGexMgAAuhMrfwNAeElPT9d7772n/fv3q7i4uMVe5BEjRmjDhg366KOP9Pe//1033nhjt/c4dxShOgSGsgI4AAA9wjenemgi060AIBzMnz9fVqtVo0aNUnJycotzpJcvX66EhARddNFFysrKUmZmpi644IIerm3bhP/4gD7Id7e84DihGgCA7uL1mjpwnOHfABBOzjzzTG3fvj3g2KxZs5qVS09P15tvvhlwbO7cuQHPTx4OHmxrrxMnTnSonu1BT3UIpDH8GwCAbldUVq0aj1dRFoOFQQEA3YZQHQK+u+VFpW5V1556kj4AAGi//KP1N6+HJLhktRghrg0AoK8iVIdAQrRN/ez1y877hqUBANCTVq5cqfT0dDmdTk2YMEE7duxoseyll14qwzCaPa666ip/GdM0tWTJEg0aNEgul0sZGRn64osveuKjtKiAod8AgB5AqA4BwzD8Q8CZVw0A6Gnr169XTk6Oli5dql27dmn06NHKzMzU4cOHg5bfsGGDDh065H98/PHHslqtuv766/1lHnroIT322GNatWqV3nvvPfXr10+ZmZmqrq7uqY/VjH+PakI1AKAbEapDxB+qmVcNAOhhy5cv15w5czR79myNGjVKq1atUnR0tFavXh20fGJiolJT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e71eMzU11fzVr37lP3bixAnT4XCY//3f/x2CGqIvo70/Ndr7RrT1gWjrg6Otb462PlBE9FTX1NRo586dysjI8B+zWCzKyMjQ9u3bQ1iz8FJSUiJJSkxMDHFNwsPcuXN11VVXBfx3E8k2btyocePG6frrr9fAgQN1/vnn6+mnnw51tULuoosuUl5enj7//HNJ0t///ndt27ZNkydPDnHNwsdXX32lwsLCgP8vxcXFacKECfwbjC5Fe982tPeNaOsD0dYHR1t/apHe1keFugI9obi4WB6PRykpKQHHU1JStHv37hDVKrx4vV7NmzdPF198sc4555xQVyfkXnzxRe3atUvvv/9+qKsSNr788ks9+eSTysnJ0V133aX3339f//mf/ym73a6ZM2eGunohs2DBApWWluqss86S1WqVx+PR/fffr+nTp4e6amGjsLBQkoL+G+x7DegKtPenRnvfiLa+Odr64GjrTy3S2/qICNU4tblz5+rjjz/Wtm3bQl2VkCsoKNDtt9+uLVu2yOl0hro6YcPr9WrcuHF64IEHJEnnn3++Pv74Y61atSqiG9qXXnpJa9eu1bp16/Ttb39bH330kebNm6fBgwdH9PcCIDzR3tejrQ+Otj442nqcSkQM/05KSpLValVRUVHA8aKiIqWmpoaoVuEjOztb//d//6e33npLQ4cODXV1Qm7nzp06fPiwLrjgAkVFRSkqKkpvv/22HnvsMUVFRcnj8YS6iiExaNAgjRo1KuDY2Wefrfz8/BDVKDz87Gc/04IFC3TDDTfo3HPP1Y9//GPdcccdys3NDXXVwobv31n+DUZ3o71vHe19I9r64Gjrg6OtP7VIb+sjIlTb7XaNHTtWeXl5/mNer1d5eXmaOHFiCGsWWqZpKjs7W6+++qrefPNNDR8+PNRVCguXX365/vnPf+qjjz7yP8aNG6fp06fro48+ktVqDXUVQ+Liiy9utgXL559/rm9961shqlF4qKyslMUS+E+p1WqV1+sNUY3Cz/Dhw5Wamhrwb3Bpaanee++9iP43GF2P9j442vvmaOuDo60Pjrb+1CK9rY+Y4d85OTmaOXOmxo0bp/Hjx2vFihWqqKjQ7NmzQ121kJk7d67WrVun//mf/1H//v398x3i4uLkcrlCXLvQ6d+/f7N5Zv369dOAAQMiev7ZHXfcoYsuukgPPPCAfvSjH2nHjh166qmn9NRTT4W6aiGVlZWl+++/X8OGDdO3v/1tffjhh1q+fLluvvnmUFetR5WXl2vv3r3+51999ZU++ugjJSYmatiwYZo3b55+8YtfaMSIERo+fLgWL16swYMH65prrgldpdEn0d43R3vfHG19cLT1wdHW16Otb0Wolx/vSY8//rg5bNgw0263m+PHjzfffffdUFcppCQFfTz33HOhrlrYYZuNev/7v/9rnnPOOabD4TDPOuss86mnngp1lUKutLTUvP32281hw4aZTqfTPO2008y7777bdLvdoa5aj3rrrbeC/nsyc+ZM0zTrt9pYvHixmZKSYjocDvPyyy839+zZE9pKo8+ivQ9Ee982tPX1aOubo62vR1vfMsM0TbMnQzwAAAAAAH1FRMypBgAAAACgOxCqAQAAAADoIEI1AAAAAAAdRKgGAAAAAKCDCNUAAAAAAHQQoRoAAAAAgA4iVAMAAAAA0EGEagAAAAAAOohQDQAAAABABxGqAQAAAADoIEI1AAAAAAAdRKgGAAAAAKCD/j/0vtWCjgKJNAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "깨끗한 테스트 세트 정확도: 100.00%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 7] “셀 평균 intensity로 0/1 인식” 함수"
      ],
      "metadata": {
        "id": "nGFomJQhUytX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def recognize_bits_from_image(\n",
        "    img,\n",
        "    cell_size=CELL_SIZE,\n",
        "    intensity_threshold=0.5\n",
        "):\n",
        "    \"\"\"\n",
        "    셀 평균 intensity >= threshold -> 1, < threshold -> 0 으로 인식\n",
        "    (라이다가 관측한 NIR 평균 강도 기준의 디지털 인식)\n",
        "    \"\"\"\n",
        "    if img.ndim == 3:\n",
        "        img = img[..., 0]\n",
        "\n",
        "    recognized_bits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)\n",
        "    cell_intensity  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)\n",
        "\n",
        "    for r in range(GRID_SIZE):\n",
        "        for c in range(GRID_SIZE):\n",
        "            y_start = r * cell_size\n",
        "            y_end   = (r + 1) * cell_size\n",
        "            x_start = c * cell_size\n",
        "            x_end   = (c + 1) * cell_size\n",
        "\n",
        "            cell = img[y_start:y_end, x_start:x_end]\n",
        "            mean_val = cell.mean()\n",
        "            cell_intensity[r, c] = mean_val\n",
        "            recognized_bits[r, c] = 1 if mean_val >= intensity_threshold else 0\n",
        "\n",
        "    return recognized_bits, cell_intensity\n"
      ],
      "metadata": {
        "id": "GmBAUAabSZTz"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 8] 부분 가려짐(Partial Occlusion) 시뮬레이터"
      ],
      "metadata": {
        "id": "Xo3cf7gPU14k"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def generate_partially_occluded_pattern_image(\n",
        "    pattern_id,\n",
        "    img_size=IMG_SIZE,\n",
        "    cell_size=CELL_SIZE,\n",
        "    high_intensity=0.9,\n",
        "    low_intensity=0.2,\n",
        "    noise_level=0.08,\n",
        "    blur=False,\n",
        "    min_coverage=0.0,\n",
        "    max_coverage=0.8\n",
        "):\n",
        "    \"\"\"\n",
        "    1) 기본 3x3 NIR 패턴 이미지 생성\n",
        "    2) 각 셀 별로 [min_coverage, max_coverage] 범위에서 랜덤하게\n",
        "       '가려진 면적 비율'을 정하고, 그 비율만큼 픽셀을 0으로 만든다.\n",
        "    3) 셀별 실제 가려진 비율(coverage_map)과 최종 평균 intensity(intensity_map)를 반환.\n",
        "    \"\"\"\n",
        "    base_img = generate_pattern_image(\n",
        "        pattern_id,\n",
        "        img_size=img_size,\n",
        "        cell_size=cell_size,\n",
        "        high_intensity=high_intensity,\n",
        "        low_intensity=low_intensity,\n",
        "        noise_level=noise_level,\n",
        "        blur=blur\n",
        "    )\n",
        "    img = base_img.copy()\n",
        "\n",
        "    coverage_map  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)\n",
        "    intensity_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)\n",
        "\n",
        "    for r in range(GRID_SIZE):\n",
        "        for c in range(GRID_SIZE):\n",
        "            y_start = r * cell_size\n",
        "            y_end   = (r + 1) * cell_size\n",
        "            x_start = c * cell_size\n",
        "            x_end   = (c + 1) * cell_size\n",
        "\n",
        "            cell = img[y_start:y_end, x_start:x_end]\n",
        "\n",
        "            target_coverage = np.random.uniform(min_coverage, max_coverage)\n",
        "            mask = np.random.rand(*cell.shape) < target_coverage  # True = 가려짐\n",
        "\n",
        "            # 가려진 부분은 NIR 완전 차단 가정 → intensity = 0\n",
        "            cell[mask] = 0.0\n",
        "            img[y_start:y_end, x_start:x_end] = cell\n",
        "\n",
        "            coverage_map[r, c] = mask.mean()     # 실제 가려진 비율\n",
        "            intensity_map[r, c] = cell.mean()    # 최종 평균 intensity\n",
        "\n",
        "    img = np.clip(img, 0.0, 1.0)\n",
        "    return img, coverage_map, intensity_map\n"
      ],
      "metadata": {
        "id": "8AzJvS1oSZWf"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 9] 한 패턴에 대해 “원래 비트 / 가려진 비율 / intensity / 인식 결과” 보기"
      ],
      "metadata": {
        "id": "D9a2ZLOEU4NS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def simulate_one_pattern_with_partial_occlusion(\n",
        "    pattern_id,\n",
        "    intensity_threshold=0.5,\n",
        "    min_coverage=0.0,\n",
        "    max_coverage=0.8\n",
        "):\n",
        "    original_bits = bits_to_grid(id_to_pattern_bits(pattern_id))\n",
        "    img, coverage_map, intensity_map = generate_partially_occluded_pattern_image(\n",
        "        pattern_id,\n",
        "        min_coverage=min_coverage,\n",
        "        max_coverage=max_coverage\n",
        "    )\n",
        "    recognized_bits, _ = recognize_bits_from_image(\n",
        "        img,\n",
        "        intensity_threshold=intensity_threshold\n",
        "    )\n",
        "\n",
        "    plt.figure(figsize=(4,4))\n",
        "    plt.imshow(img, cmap=\"gray\")\n",
        "    plt.title(f\"Pattern ID {pattern_id}\\nthreshold={intensity_threshold}\")\n",
        "    plt.axis(\"off\")\n",
        "    plt.show()\n",
        "\n",
        "    print(\"원래 비트 (3x3):\")\n",
        "    print(original_bits)\n",
        "    print(\"\\n셀별 실제 가려진 비율 (0~1):\")\n",
        "    print(coverage_map)\n",
        "    print(\"\\n가려진 후 셀 평균 intensity:\")\n",
        "    print(intensity_map)\n",
        "    print(\"\\n임계값 기반 인식 결과 (3x3):\")\n",
        "    print(recognized_bits)\n",
        "    print(\"\\n틀린 셀 개수:\", (original_bits != recognized_bits).sum(), \"/ 9\")\n",
        "\n",
        "    return original_bits, coverage_map, intensity_map, recognized_bits\n",
        "\n",
        "\n",
        "_ = simulate_one_pattern_with_partial_occlusion(\n",
        "    pattern_id=5,\n",
        "    intensity_threshold=0.5,\n",
        "    min_coverage=0.0,\n",
        "    max_coverage=0.8\n",
        ")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 771
        },
        "id": "QM8aoge8SZZP",
        "outputId": "f6376809-fb32-4dc1-d2a7-5e55cc4e7bec"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 400x400 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "원래 비트 (3x3):\n",
            "[[0 0 0]\n",
            " [0 0 0]\n",
            " [1 0 1]]\n",
            "\n",
            "셀별 실제 가려진 비율 (0~1):\n",
            "[[0.03 0.57 0.03]\n",
            " [0.39 0.69 0.76]\n",
            " [0.66 0.29 0.05]]\n",
            "\n",
            "가려진 후 셀 평균 intensity:\n",
            "[[0.18497528 0.0728218  0.1897572 ]\n",
            " [0.12383486 0.0712871  0.05155854]\n",
            " [0.3088407  0.14177853 0.8583117 ]]\n",
            "\n",
            "임계값 기반 인식 결과 (3x3):\n",
            "[[0 0 0]\n",
            " [0 0 0]\n",
            " [0 0 1]]\n",
            "\n",
            "틀린 셀 개수: 1 / 9\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 10] 통계용: 여러 패턴/셀에 대해 coverage–intensity–정답 여부 수집"
      ],
      "metadata": {
        "id": "aDXwQ3I8U8QV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def collect_partial_occlusion_stats(\n",
        "    num_samples=2000,\n",
        "    intensity_threshold=0.5,\n",
        "    min_coverage=0.0,\n",
        "    max_coverage=0.8\n",
        "):\n",
        "    coverages = []\n",
        "    intensities = []\n",
        "    correct_flags = []\n",
        "\n",
        "    for _ in range(num_samples):\n",
        "        pattern_id = np.random.randint(0, N_CLASSES)\n",
        "        img, coverage_map, intensity_map = generate_partially_occluded_pattern_image(\n",
        "            pattern_id,\n",
        "            min_coverage=min_coverage,\n",
        "            max_coverage=max_coverage\n",
        "        )\n",
        "\n",
        "        original_bits = bits_to_grid(id_to_pattern_bits(pattern_id))\n",
        "        recognized_bits, _ = recognize_bits_from_image(\n",
        "            img,\n",
        "            intensity_threshold=intensity_threshold\n",
        "        )\n",
        "\n",
        "        for r in range(GRID_SIZE):\n",
        "            for c in range(GRID_SIZE):\n",
        "                if original_bits[r, c] == 1:  # 원래 고반사 셀만 분석\n",
        "                    cov = coverage_map[r, c]\n",
        "                    inten = intensity_map[r, c]\n",
        "                    correct = 1 if recognized_bits[r, c] == 1 else 0\n",
        "\n",
        "                    coverages.append(cov)\n",
        "                    intensities.append(inten)\n",
        "                    correct_flags.append(correct)\n",
        "\n",
        "    coverages = np.array(coverages, dtype=np.float32)\n",
        "    intensities = np.array(intensities, dtype=np.float32)\n",
        "    correct_flags = np.array(correct_flags, dtype=np.int32)\n",
        "\n",
        "    print(\"수집된 셀 샘플 수(원래 1인 셀):\", len(coverages))\n",
        "    return coverages, intensities, correct_flags\n",
        "\n",
        "\n",
        "coverages, intensities, correct_flags = collect_partial_occlusion_stats(\n",
        "    num_samples=2000,\n",
        "    intensity_threshold=0.5,\n",
        "    min_coverage=0.0,\n",
        "    max_coverage=0.8\n",
        ")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8AAiuwnnSZbx",
        "outputId": "ce45a746-076a-4f48-de2f-5d3f917b7f25"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "수집된 셀 샘플 수(원래 1인 셀): 5022\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 11] Coverage vs 인식 정확도 / 평균 intensity 그래프"
      ],
      "metadata": {
        "id": "dvQnBSC0U-lQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_accuracy_vs_coverage(\n",
        "    coverages,\n",
        "    intensities,\n",
        "    correct_flags,\n",
        "    num_bins=10\n",
        "):\n",
        "    bins = np.linspace(0.0, 1.0, num_bins + 1)\n",
        "    bin_indices = np.digitize(coverages, bins) - 1  # 0~num_bins-1\n",
        "    bin_centers = 0.5 * (bins[:-1] + bins[1:])\n",
        "\n",
        "    acc_per_bin = np.zeros(num_bins)\n",
        "    mean_intensity_per_bin = np.zeros(num_bins)\n",
        "    count_per_bin = np.zeros(num_bins, dtype=int)\n",
        "\n",
        "    for i in range(num_bins):\n",
        "        in_bin = bin_indices == i\n",
        "        if in_bin.sum() > 0:\n",
        "            count_per_bin[i] = in_bin.sum()\n",
        "            acc_per_bin[i] = correct_flags[in_bin].mean()\n",
        "            mean_intensity_per_bin[i] = intensities[in_bin].mean()\n",
        "        else:\n",
        "            acc_per_bin[i] = np.nan\n",
        "            mean_intensity_per_bin[i] = np.nan\n",
        "\n",
        "    plt.figure(figsize=(12,4))\n",
        "\n",
        "    # (1) coverage vs accuracy\n",
        "    plt.subplot(1,2,1)\n",
        "    plt.plot(bin_centers * 100, acc_per_bin * 100, marker='o')\n",
        "    plt.xlabel(\"가려진 면적 (%, coverage)\")\n",
        "    plt.ylabel(\"인식 정확도 (%)\")\n",
        "    plt.ylim(0, 105)\n",
        "    plt.title(\"가려진 면적 vs 고반사 셀(1) 인식 정확도\")\n",
        "\n",
        "    # (2) coverage vs mean intensity\n",
        "    plt.subplot(1,2,2)\n",
        "    plt.plot(bin_centers * 100, mean_intensity_per_bin, marker='o')\n",
        "    plt.xlabel(\"가려진 면적 (%, coverage)\")\n",
        "    plt.ylabel(\"셀 평균 intensity (0~1)\")\n",
        "    plt.ylim(0, 1.0)\n",
        "    plt.title(\"가려진 면적 vs 셀 평균 intensity\")\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    # 표 형식으로도 확인\n",
        "    print(\"구간별 통계: coverage[%], n, acc[%], mean_I\")\n",
        "    for i in range(num_bins):\n",
        "        print(\n",
        "            f\"{bins[i]*100:5.1f}~{bins[i+1]*100:5.1f} \"\n",
        "            f\"n={count_per_bin[i]:4d}, \"\n",
        "            f\"acc={acc_per_bin[i]*100:6.2f}, \"\n",
        "            f\"mean_I={mean_intensity_per_bin[i]:.3f}\"\n",
        "        )\n",
        "\n",
        "\n",
        "plot_accuracy_vs_coverage(coverages, intensities, correct_flags, num_bins=10)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "5b25t8ACSqLh",
        "outputId": "deba4faf-9c1e-4295-de9a-75055216345d"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 44032 (\\N{HANGUL SYLLABLE GA}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 47140 (\\N{HANGUL SYLLABLE RYEO}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 51652 (\\N{HANGUL SYLLABLE JIN}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 47732 (\\N{HANGUL SYLLABLE MYEON}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 51201 (\\N{HANGUL SYLLABLE JEOG}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 51064 (\\N{HANGUL SYLLABLE IN}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 49885 (\\N{HANGUL SYLLABLE SIG}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 51221 (\\N{HANGUL SYLLABLE JEONG}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 54869 (\\N{HANGUL SYLLABLE HWAG}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 46020 (\\N{HANGUL SYLLABLE DO}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 44256 (\\N{HANGUL SYLLABLE GO}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 48152 (\\N{HANGUL SYLLABLE BAN}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 49324 (\\N{HANGUL SYLLABLE SA}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 49472 (\\N{HANGUL SYLLABLE SEL}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 54217 (\\N{HANGUL SYLLABLE PYEONG}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/tmp/ipython-input-840223333.py:43: UserWarning: Glyph 44512 (\\N{HANGUL SYLLABLE GYUN}) missing from font(s) DejaVu Sans.\n",
            "  plt.tight_layout()\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51064 (\\N{HANGUL SYLLABLE IN}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49885 (\\N{HANGUL SYLLABLE SIG}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51221 (\\N{HANGUL SYLLABLE JEONG}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54869 (\\N{HANGUL SYLLABLE HWAG}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 46020 (\\N{HANGUL SYLLABLE DO}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44032 (\\N{HANGUL SYLLABLE GA}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 47140 (\\N{HANGUL SYLLABLE RYEO}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51652 (\\N{HANGUL SYLLABLE JIN}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 47732 (\\N{HANGUL SYLLABLE MYEON}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51201 (\\N{HANGUL SYLLABLE JEOG}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44256 (\\N{HANGUL SYLLABLE GO}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 48152 (\\N{HANGUL SYLLABLE BAN}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49324 (\\N{HANGUL SYLLABLE SA}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49472 (\\N{HANGUL SYLLABLE SEL}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54217 (\\N{HANGUL SYLLABLE PYEONG}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44512 (\\N{HANGUL SYLLABLE GYUN}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x400 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABKUAAAGGCAYAAACqvTJ0AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAjvJJREFUeJzs3Xl8TOcex/HPmexIYs1GSKgt1lhC7DRKq1qlLUot1V1bqovqQt0uuldbpa0uaGuptpS2uBpbEWJplNpJCLIgJBGyyMz9Q+U2FSQkOcnk+3695nWvM8+Z+T5D+eU3z3mOYbPZbIiIiIiIiIiIiBQji9kBRERERERERESk7FFTSkREREREREREip2aUiIiIiIiIiIiUuzUlBIRERERERERkWKnppSIiIiIiIiIiBQ7NaVERERERERERKTYqSklIiIiIiIiIiLFTk0pEREREREREREpdmpKiYiIiIiIiIhIsVNTSkTERFarlcaNG/Paa69d0/lt27bl2WefLeRUIiIiIiXXyy+/jGEYZscoFDNmzMAwDGJiYsyOImIKNaVEREw0Z84cYmNjeeyxx3KOnTlzhgkTJtCzZ08qV66MYRjMmDEjz/PHjh3Lxx9/THx8fDElFhERESm91q9fz8svv8zp06fNjnJZU6dOvWztJ2Jv1JQSKWZ//fUXzs7OVKhQIc+Hs7MzBw4cKPRxl+Pj43PZc11dXfnyyy8LNM4Mbdu2pXz58nlmc3NzY8KECUUyLi9jx47Fzc0tz3PLly9Ply5dco1/++23GTBgAJ6enjnHTpw4wX/+8x927dpFs2bNrjj322+/HQ8PD6ZOnVrsn9P1zL+wx4mISNmkuqrwlaQ64HJefPFFzp07d03zW79+PRMnTiwxTal7772Xc+fOUatWrZxjakpJWaKmlEgxs9lshISEcObMmTwfLVq0wGazFfq4yzl//jynT5/O89zRo0djtVoLNM4M58+fZ9u2bXlme//998nOzi6ScXnJzs7mo48+yvPcLVu2cP78+Zyxf/zxB9u2bePuu+/O9Rq+vr7ExcVx6NAh3n777SvO3WKxcOeddzJr1qwr/j6XtPkX9jgRESmbVFcVvpJUB1yOo6Mjrq6uhTpvszg4OODq6mo3lyOKFJSaUiJS5Bo3bkzXrl0vOW61WqlevTp33nlnzrG5c+fSsmVL3N3d8fDwoEmTJnzwwQfFGbfYLFy4EGdnZzp16pTruIuLCz4+Pvl+ne7du3Po0CGioqIKOaGIiIiUNKqr8t5TyjAMHnvsMRYuXEjjxo1xcXGhUaNGLF26NNd5zzzzDACBgYEYhnHJfk7ffPMNLVu2xM3NjcqVKzNgwABiY2NzvVeXLl1o3LgxO3fupGvXrpQrV47q1avz1ltvXZL1o48+olGjRpQrV45KlSrRqlUrZs+enfP8v/eUCggI4K+//mL16tU5+bp06cLBgwcxDIP333//kvdYv349hmEwZ86cAn+WImZTU0pEilz//v1Zs2bNJfserV27lmPHjjFgwAAAli9fzsCBA6lUqRJvvvkmb7zxBl26dGHdunVmxC5y69evp3Hjxjg5OV3X67Rs2RLAbj8nERER+T/VVZe3du1aHn30UQYMGMBbb71Feno6/fr14+TJkwD07duXgQMHAvD+++/z9ddf8/XXX1OtWjUAXnvtNYYMGULdunV57733GD16NOHh4XTq1OmSy/1OnTpFz549adasGe+++y4NGjRg7NixLFmyJGfM9OnTeeKJJwgKCmLy5MlMnDiR5s2bs3HjxsvOYfLkydSoUYMGDRrk5HvhhReoXbs27du359tvv73knG+//RZ3d3duv/326/0IRYqdo9kBRMT+9e/fn/Hjx/P999/n2tB73rx5VKhQgV69egHwyy+/4OHhwbJly3BwcDArbrHZvXs3bdq0ue7XqV69Os7OzuzcubMQUomIiEhJprrq8nbt2sXOnTupU6cOAF27dqVZs2bMmTOHxx57jKZNm9KiRQvmzJlDnz59CAgIyDn30KFDTJgwgVdffZXnn38+53jfvn0JDg5m6tSpuY4fO3aMWbNmce+99wIwYsQIatWqxRdffMHNN98MXPg9aNSoEfPnz8/3HPr06cOLL75I1apVGTx4cK7nhgwZwkMPPcTu3btp0KABAFlZWXz33Xf07duXcuXKFewDEykBtFJKRIpcvXr1aN68OfPmzcs5lp2dzffff0/v3r1xc3MDoGLFiqSlpbF8+XKzoharkydPUqlSpUJ5rUqVKnHixIlCeS0REREpuVRXXV5YWFhOQwqgadOmeHh4cPDgwaue++OPP2K1Wrn77rs5ceJEzsPHx4e6deuycuXKXOMrVKiQq2nk7OxMSEhIrveqWLEiR44cYdOmTYUwO7j77rtx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ff+1N4KIyPVYuXKl2RFERExnGAZd63vRtb4X22JP89magyzZEceavcdZs/c4jat78FCnOtzc2AdHhyK9wEdETFSglVL9+vUjLi4u3y8eFBTE559/fk3BipO+8RMRgKU74nj4m62XHL/4Hd20wS3UmBIpw1QvXKDPQUSKyqGTaXyxNprvNseSnmUFoEYlN+7vEMjdrf0p56wtkUVKi/zWCwVqSgUHB/PHH3/kO0RISAiRkZH5Hm8WFVcikm210eHNFbnuHvNPBhfuJrN2bDddyidSRhVmvfD777/z6aefcvDgQebPn0/16tX5+uuvCQwMpEOHDoWUuGiobhKRopaUlsmsiBhmro/h1NksACqWc2JI21oMaRdA1QouJicUkavJb71QoHWQuqZXROxVZHTSZRtSADYgLjmdyOiky44REcmPH374gR49euDm5sbWrVvJyMgAIDk5mddff93kdCIi5qtc3pnRYfVY/9yNvHJ7I2pWLsfps1l8uGI/7d9YwQsLthNzIs3smCJSCHRxrogIkJh6+YbUtYwTEbmcV199lU8++YTp06fj5OSUc7x9+/Zs3XrpJcQiImWVm7MD94YGsPLpLnx8Twua1fAk47yVbzcepuu7q3jkmy38cfiU2TFF5DroolwREcDL3bVQx4mIXM6ePXvo1KnTJcc9PT05ffp08QcSESnhHCwGvZr6cksTHzYcTOKzNQdYuec4S3bEs2RHPCGBlXmoU2261vfCom0WREoVNaVERICQwMr4eroSn5xOXhvtXdxTKiSwcnFHExE74+Pjw/79+wkICMh1fO3atdSuXducUCIipYBhGITWqUJonSrsiU/lszUHWbTtKJHRSURGJ1HXqwIPdKrN7c39cHF0MDuuiORDgZpSaWlp3HffffkaW4D900VETOdgMZjQO4hHvtmKAXk2pib0DtIm5yJy3R544AFGjRrFl19+iWEYHDt2jIiICJ5++mleeukls+OJiJQK9X3ceffuZjzdox5frYth9sbD7Es8w7Pf/8m7/93D8PaB3NOmJh6uTld/MRExTYHuvnfw4EGysrLy/eJubm7UrFnzmoIVJ91FRkQuWrojjomLd+ba9NxiwAcDgundzM/EZCJitsKqF2w2G6+//jqTJk3i7NmzALi4uPD000/zyiuvFFbcIqO6SURKopT0LGZvPMxX66JJSLlwA4kK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N8vcxNwkl/E+i7lJPp9TT/wbchPA7NSs1l5pnV5PXV2dKCoqEkVFRQKA2LhxoygqKhLXrl0TQjzeUtLW1lYcOnRIlJSUiEmTJrXbLSUXLVokbGxsRG5urt7WoVqtVmqzcOFCoVQqxcmTJ8Wvv/4qhg8fLoYPH96KvX51sbGxIi8vT1RVVYmSkhIRGxsrFAqFyM7OFkLIa6zGPL2LjBDyGm9kZKTIzc0VVVVV4syZM2LcuHGiS5cuorq6Wgghr7H+8ssvwszMTKjVanH58mWh0WiEtbW1+P7776U2cvqcEkKIR48eCaVSKWJiYgzq5HRvQ0JCRPfu3aVtjX/88UfRpUsXER0dLbWR272l9o+5iblJDmM1hrlJHmNlbtInp3srBLNTczgp1c6dOnVKADD4CQkJEUI83lZSpVIJBwcHYWFhIQICAkR5eXnrdvoVGRsnAJGWlia1uX//vli8eLGws7MT1tbWYvLkyeLGjRut1+nXMHfuXOHs7CzMzc1F165dRUBAgBSshJDXWI15NlzJabzTp08Xjo6OwtzcXHTv3l1Mnz5dXLlyRaqX01iFEOLw4cOif//+wsLCQvTt21ds375dr15On1NCCHHs2DEBwOgY5HRv7969KyIiIoRSqRSWlpaiV69eIj4+XjQ2Nkpt5HZvqf1jbmJuksNYjWFuksdYhWBueprc7i2zk3EKIYRoqaeyiIiIiIiIiIiIAK4pRURERERERERErYCTUkRERERERERE1OI4KUVERERERERERC2Ok1JERERERERERNTiOClFREREREREREQtjpNSRERERERERETU4jgpRURERERERERELY6TUkRERERERERE1OI4KUVEspCamorx48e3djdkr6amBvb29vjrr79auytERET0ipibWgZzE9GLKYQQorU7QUQtKy8vDwsWLIClpaVeuU6nw5gxY7B582YMGzYMjY2NBsfW19fjjz/+wKZNm7B7926YmZnp1T948ADx8fEIDg42OHby5MmoqqoyKNdqtfjpp59w7tw5qNVqmJub69U/fPgQs2bNQkxMjNHxNDQ0oFevXti/fz/8/PwAAMePH8dnn32Gf/75B5MmTUJqaqp03jt37sDHxwfHjx+Hs7Pzc94pMiYqKgq3b99Gampqa3eFiIjojWNuYm56HcxNRM9n9uImRCQ39+/fx4wZM/Dll1/q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3oKe6enpyMrKwkcffYQVK1bAx8cHAHD16lUoFArk5uY2ey2VSoXIyEgkJCTAw8MD06dPR3V1NQDA2toax44dw61bt+Dj44Np06YhICDAYFFQNzc3+Pr6oqSkxOCbtQEDBiAvLw8VFRUYNWoUBg0ahISEBL3+GzNx4kQsX74c4eHhGDhwIAoKCqTdZZ4IDg5GXFwcoqKiMHjwYFRVVSE0NFRvXYmwsDDs3LkTaWlp8PLywpgxY5Ceng4XFxepzaFDh6BUKjFq1Kjn9omIiIjaHuYm5iaitoT/vkdE7V6/fv0QFBSErVu3Ii4uTq9uz549Bu19fX1RVlZmUF5VVQVbW1t4e3s3ey0TExPEx8frPe79NC8vL2kBzOc5f/58s3U+Pj7Izs5utv7q1atGy5OSkpCUlKRXtmzZMr3XKpVKL3S9//77Bo+8z5w5EzNnzmz2+snJyUhISGi2noiIiNou5qbHmJuI2gZOShGRLKxfvx6HDx9+rXMcPXoUK1euhJ2d3f+pV22LVqvFd999h8DAQJiammLPnj04ceIEjh8//tLnqKmpwZQpU/DJJ5+8wZ4SERHRm8Tc9GLMTUQtg5NSRP9CNjY2OHLkCI4cOWJQFxgYCACwtbU1un0u8Phbrx49eiAqKspo/cqVK42We3h4NHtOKysr2NvbY+3atQaPbQNAaGio0eOe6NmzJ5YsWfLcNi+yfv361zq+rVMoFDh69CjUajUaGhrg7u6OAwcOYNy4cS99ji5duiA6OvoN9pKIiKhtYW4yjrnpxZibiF5MIYQQrd0JIiIiIiIiIiL6d+FC50RERERERERE1OI4KUVERERERERERC2Ok1JERERERERERNTiOClFREREREREREQtjpNSRERERERERETU4jgpRURERERERERELY6TUkRERERERERE1OI4KUVERERERERERC2Ok1JERERERERERNTi/gO896KLZMnGtwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "구간별 통계: coverage[%], n, acc[%], mean_I\n",
            "  0.0~ 10.0 n= 626, acc=100.00, mean_I=0.856\n",
            " 10.0~ 20.0 n= 657, acc=100.00, mean_I=0.765\n",
            " 20.0~ 30.0 n= 612, acc=100.00, mean_I=0.677\n",
            " 30.0~ 40.0 n= 616, acc=100.00, mean_I=0.586\n",
            " 40.0~ 50.0 n= 643, acc= 48.37, mean_I=0.499\n",
            " 50.0~ 60.0 n= 594, acc=  0.00, mean_I=0.409\n",
            " 60.0~ 70.0 n= 714, acc=  0.00, mean_I=0.312\n",
            " 70.0~ 80.0 n= 458, acc=  0.00, mean_I=0.227\n",
            " 80.0~ 90.0 n= 102, acc=  0.00, mean_I=0.161\n",
            " 90.0~100.0 n=   0, acc=   nan, mean_I=nan\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Bwr_P54wqQMn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[셀 12] Coverage–Intensity 평면 산점도 (정답/오답 구분)"
      ],
      "metadata": {
        "id": "6EaCliALVCHm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_scatter_coverage_intensity(\n",
        "    coverages,\n",
        "    intensities,\n",
        "    correct_flags\n",
        "):\n",
        "    plt.figure(figsize=(6,5))\n",
        "    correct_idx = correct_flags == 1\n",
        "    wrong_idx = ~correct_idx\n",
        "\n",
        "    plt.scatter(coverages[correct_idx]*100,\n",
        "                intensities[correct_idx],\n",
        "                alpha=0.4, s=15, label=\"정답(1로 인식)\")\n",
        "    plt.scatter(coverages[wrong_idx]*100,\n",
        "                intensities[wrong_idx],\n",
        "                alpha=0.6, s=15, label=\"오답\")\n",
        "\n",
        "    plt.xlabel(\"가려진 면적 (%, coverage)\")\n",
        "    plt.ylabel(\"셀 평균 intensity (0~1)\")\n",
        "    plt.title(\"Coverage–Intensity 평면에서의 인식 결과\")\n",
        "    plt.grid(True)\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "\n",
        "plot_scatter_coverage_intensity(coverages, intensities, correct_flags)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 779
        },
        "id": "b3tMyTrSSsbx",
        "outputId": "79ff9ca7-fd2d-4c0a-a8ef-ac29ab6439b0"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 50640 (\\N{HANGUL SYLLABLE E}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49436 (\\N{HANGUL SYLLABLE SEO}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51032 (\\N{HANGUL SYLLABLE YI}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44208 (\\N{HANGUL SYLLABLE GYEOL}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44284 (\\N{HANGUL SYLLABLE GWA}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 45813 (\\N{HANGUL SYLLABLE DAB}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 47196 (\\N{HANGUL SYLLABLE RO}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n",
            "/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 50724 (\\N{HANGUL SYLLABLE O}) missing from font(s) DejaVu Sans.\n",
            "  fig.canvas.print_figure(bytes_io, **kw)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhgAAAHWCAYAAAA1jvBJAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAbs9JREFUeJzt3Xd0VNXax/HvpCeQCiT0hNA7CIKACgKKXhWxXOGKUuwigqJSVIoi9oLdV0WxoCDXa0NEEQFFEBREpbeEnoSQkJ7MZOa8f4wZGBIgyZwwKb/PWlmHOWWfJ1skT3a1GIZhICIiImIiH28HICIiItWPEgwRERExnRIMERERMZ0SDBERETGdEgwRERExnRIMERERMZ0SDBERETGdEgwRERExnRIMERERMZ0SDBGpUhITE7FYLMydO9fboYjIaSjBkEpl9+7d3HHHHcTHxxMUFERYWBh9+vThpZdeIi8vz9vhVVkzZszAYrGQmppa5mcPHTrEjBkz2Lhxo/mBmWTx4sXMmDHD22GIyAmUYEil8c0339CxY0c+/fRTrrzySl555RWefPJJmjZtyoMPPsj48eO9HWKNdOjQIR599NFKk2DExsaSl5fHTTfd5Dq3ePFiHn30UVPfM3ToUEJCQqhdu3axr5CQEEaOHFkl7ivJ66+/TlBQUInP1q5dm7i4uDLdV5LNmzcTEBBwymcDAgLYvXt3qe+TqkcJhlQKCQkJDBs2jNjYWLZs2cJLL73Ebbfdxt13380nn3zCli1baN++vbfDLFFubq63Q6hRLBYLQUFB+Pr6Vuh77HY7X331FdnZ2cW+/ve//2G326vEfSVxOBw88MADJT6bmppKYWFhme4riWEY9OjRo8Rns7OzOeecczAMo9T3SdWjBEMqhWeeeYbs7GzmzJlDgwYNil1v0aKFWwtGYWEhM2fOpHnz5gQGBhIXF8dDDz1EQUGB654rrriC+Pj4Et/Xq1cvunfv7nbuo48+olu3bgQHBxMVFcWwYcPYv3+/2z39+vWjQ4cOrF+/ngsvvJCQkBAeeughAL788ksuv/xyGjZsSGBgIM2bN2fmzJkl/kP/2muvER8fT3BwMD169ODnn3+mX79+9OvXz+2+goICpk+fTosWLQgMDKRJkyZMnDjR7fssr6LvZcuWLVx00UWEhITQqFEjnnnmGdc9K1as4NxzzwVg9OjRWCyWYuMf1q5dy6WXXkp4eDghISH07duXX375xe1dRV00u3btYtSoUURERBAeHs7o0aOLJWhLly7l/PPPJyIigtq1a9O6dWtXHUPxMRijRo3itddeA3DFZ7FYMAyDuLg4rrrqqmLfe35+PuHh4dxxxx0e1aGInJoSDKkUvv76a+Lj4+ndu3ep7r/11luZNm0a55xzDi+++CJ9+/blySefZNiwYa57hg4dSkJCAr/99pvbs3v37uXXX391u3fWrFmMGDGCli1b8sILL3DvvfeybNkyLrzwQo4dO+b2/NGjR7nsssvo0qULs2fP5qKLLgJg7ty51K5dmwkTJvDSSy/RrVs3pk2bxuTJk92ef+ONNxg7diyNGzfmmWee4YILLmDIkCEcOHDA7T6Hw8HgwYN57rnnXF1GQ4YM4cUXX2To0KGlqqczSU9P59JLL6Vz5848//zztGnThkmTJvHtt98C0LZtWx577DEAbr/9dj788EM+/PBDLrzwQgB+/PFHLrzwQjIzM5k+fTpPPPEEx44do3///qxbt67Y+66//nqysrJ48sknuf7665k7d65b18bmzZu54oorKCgo4LHHHuP5559n8ODBxRKWE91xxx1cfPHFAK74PvzwQywWCzfeeCPffvstaWlpbs98/fXXZGZmcuONN3pWgSJySn7eDkAkMzOTgwcPlvibZkn+/PNP3n//fW699VbefvttAMaMGUN0dDTPPfccy5cv56KLLuKqq64iMDCQBQsWuH4LB/j000+xWCxcf/31gDPhmD59Oo8//rjbb8rXXHMNXbt25fXXX3c7n5SUxJtvvlnst9+PP/6Y4OBg1+c777yTO++8k9dff53HH3+cwMBArFYrU6dO5dxzz+XHH3/Ez8/5v2CnTp0YNWoUjRs3divvhx9+YOXKlZx//vmu8x06dODOO+9k9erVpU7ITuXQoUN88MEHrvEMt9xyC7GxscyZM4fLLruMmJgYLrvsMqZNm0avXr3cfiAbhsGdd97JRRddxLfffovFYgGcP/Dbt2/PI488wvfff+/2vq5duzJnzhzX56NHjzJnzhyefvppwNl6YbVa+fbbb6lbt26pvodevXrRqlUrli5dWixhGDFiBLNmzeLTTz/lzjvvdJ3/6KOPiIuLc6tXETGXWjDE6zIzMwEIDQ0t1f2LFy8GYMKECW7n77//fsA5WBQgLCyMyy67jE8//dStD3fBggWcd955NG3aFID//e9/OBwOrr/+elJTU11f9evXp2XLlixfvtztPYGBgYwePbpYXCcmF1lZWaSmpnLBBReQm5vLtm3bAPj99985evQot912myu5ABg+fDiRkZFu5S1cuJC2bdvSpk0bt7j69+8PUCyu8qhdu7bbD+WAgAB69OjBnj17zvjsxo0b2blzJzfccANHjx51xZeTk8OAAQP46aefcDgcbs+c+EMe4IILLuDo0aOuvwMRERGAs7vp5GfLo1WrVvTs2ZN58+a5zqWlpfHtt98yfPhwV1IkIuZTC4Z4XVhYGOD8oVwae/fuxcfHhxYtWridr1+/PhEREezdu9d1bujQoXzxxResWbOG3r17s3v3btavX8/s2bNd9+zcuRPDMGjZsmWJ7/P393f73KhRIwICAordt3nzZh555BF+/PFH1w/MIhkZGa7YgWKx+/n5FRuRv3PnTrZu3Uq9evVKjCslJQVw/sC0Wq0l3lO/fv0Szxdp3LhxsR+ykZGR/PXXX6d9rig+4LSzFTIyMtwSp6Kk7sR3gbOrJiwsjKFDh/LOO+9w6623MnnyZAYMGMA111zDddddh49P+X4fGjFiBGPHjmXv3r3ExsaycOFCbDab2ywUETGfEgzxurCwMBo2bMimTZvK9Fxpfvu88sorCQkJ4dNPP6V37958+umn+Pj48O9//9t1j8PhwGKx8O2335Y4M6F27dpun09sqShy7Ngx+vbtS1hYGI899hjNmzcnKCiIDRs2MGnSpHL9Nu5wOOjYsSMvvPBCidebNGkCOLtyVq5cWeI9Zxp9f6qZGKUZtV/0PT377LN06dKlxHtOrrszvS84OJiffvqJ5cuX880337BkyRIWLFhA//79+f7778s1c2TYsGHcd999zJs3j4ceeoiPPvqI7t2707p16zKXJSKlpwRDKoUrrriCt956izVr1tCrV6/T3hsbG4vD4WDnzp20bdvWdT45OZljx44RGxvrOlerVi2uuOIKFi5cyAsvvMCCBQu44IILaNiwoeue5s2bYxgGzZo1o1WrVuWKf8WKFRw9epT//e9/rgGQ4Jx+e3LsALt27XINDgXnrJjExEQ6derkFteff/7JgAEDTptMPf/886Snp5cr7tI41bubN28OOBPEgQMHmvY+Hx8fBgwYwIABA3jhhRd44okn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ymfqF3/zR0IqD3eGKZ/9SddmdZk5pCPRoUHeDk9ERE5j5cqV3HHHHQQFuf977XA46Nu3L6+88gq9evUiNze32GyK7OxsNm/eTGBgIAATJ07kkUcecd13+PBh7r//fn7//Xd27drFuHHjmD17tuv53bt3c9lll5U466ZZs2Z8/vnnxc7PmzePWbNmERAQ4Ha+sLCQm266iXvvvZf27duXuH5FYGAga9eupW7duowYMYLp06czZ86c0lVUOSnB8MCrP+7gu80pBPoaAGQU2PlucwoxYTt47KpOXo5OREROJy8vj2HDhjFjxgy384mJiUyePBlwzqj4+eefCQsLc5sl069fPwzD+W//qlWr2L17N9dee63rekFBAfXq1eORRx7hxRdfLPZum81G7969mTt3brFr5513XonxZmVlMXHiREaNGuV2fsWKFSxZsgTDMGjcuDErVqw4bZmjR4+mW7duPPvss0RFRZX4LjMowfDAZ+tLbmL6728H6Ne6PsmZecSEBdOhUZhaNERESiElK59NBzOr1L+f8+fP5+KLL3ZrCYmLi+Oll14C4N133/VWaCVq3749DRs25PPPP+eWW26psPdoDIYHcqxGiedzCw0++nUvSzYl8dGve3l/daLGZoiInEFKVj6frT/Aiu0p7Duax4rtKXy2/kCl//fz559/pnv37t4Oo0x69OjBzz//XKHvUIJRQXIKbGBYyCmwsXr3UVbtTPV2SCIildqmg5kczsinVUwoTaJCaBUTyuEMZ4tGZbZ3714aNmzo7TDKpGHDhuzdu7dC36EukgqyIzmLfJudIH9fagX48ef+Y1xzTmNvhyUiUmklZ+ZRK8APn39WkvSxWKgV4EdyZp6XIzu9vLy8YgNFK7vg4OAK35hOLRgVJD23kDybQXpuIQeP5ZNwNNvbIYmIVGoxYcHkWAtx/DN40mEY5FgLiQkL9nJkp1e3bl3S09O9HUaZpKWlUa9evQp9h1owzgID2HIggx+3pVSpgUsiImdTh0ZhbE/KZEdyFrUC/MixFtIgPIiOjcK9Hdppde3alS1btng7jDLZtGkT/fr1q9B3KME4S1JzC5n59WZyrIXUCvDj3PgoRvWOIymjQEmHiAgQHRrEtd0au80i6dgonHqhgd4O7bQGDRrE+++/X+z8xo0bAeeaGUeOHGHjxo0EBATQrl27sxyhu9zcXNav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Tz31FBaLhXvvvdd1TnVeMQ4ePMiNN95InTp1CA4OpmPHjvz++++u64ZhMG3aNBo0aEBwcDADBw5k586dXoy46rPb7UydOpVmzZoRHBxM8+bNmTlzptt+G6r3cjDEI/PnzzcCAgKMd99919i8ebNx2223GREREUZycrK3Q6s2Bg0aZLz33nvGpk2bjI0bNxr/+te/jKZNmxrZ2dmue+68806jSZMmxrJly4zff//dOO+884zevXt7MerqY926dUZcXJzRqVMnY/z48a7zqnPzpaWlGbGxscaoUaOMtWvXGnv27DG+++47Y9euXa57nnrqKSM8PNz44osvjD///NMYPHiw0axZMyMvL8+LkVdts2bNMurUqWMsWrTISEhIMBYuXGjUrl3beOmll1z3qN7LTgmGh3r06GHcfffdrs92u91o2LCh8eSTT3oxquotJSXFAIyVK1cahmEYx44dM/z9/Y2FCxe67tm6dasBGGvWrPFWmNVCVlaW0bJlS2Pp0qVG3759XQmG6rxiTJo0yTj//PNPed3hcBj169c3nn32Wde5Y8eOGYGBgcYnn3xyNkKsli6//HLj5ptvdjt3zTXXGMOHDzcMQ/VeXuoi8YDVamX9+vUMHDjQdc7Hx4eBAweyZs0aL0ZWvWVkZAAQFRUFwPr167HZbG7/Hdq0aUPTpk3138FDd999N5dffrlb3YLqvKJ89dVXdO/enX//+99ER0fTtWtX3n77bdf1hIQEkpKS3Oo9PDycnj17qt490Lt3b5YtW8aOHTsA+PPPP1m1ahWXXXYZoHovrxq32ZmZUlNTsdvtxMTEuJ2PiYlh27ZtXoqqenM4HNx777306dOHDh06AJCUlERAQAARERFu98bExJCUlOSFKKuH+fPns2HDBn777bdi11TnFWPPnj288cYbTJgwgYceeojffvuNcePGERAQwMiRI111W9K/Oar38ps8eTKZmZm0adMGX19f7HY7s2bNYvjw4QCq93JSgiFVyt13382mTZtYtWqVt0Op1vbv38/48eNZunQpQUFB3g6nxnA4HHTv3p0nnngCgK5du7Jp0ybefPNNRo4c6eXoqq9PP/2UefPm8fHHH9O+fXs2btzIvffeS8OGDVXvHlAXiQfq1q2Lr69vsZHzycnJ1K9f30tRVV9jx45l0aJFLF++nMaNG7vO169fH6vVyrFjx9zu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8+bGxx9/7HbPzJkzjV69ermdu++++4x+/fqdMQYRb9MYDBEpl7y8PIBiA1bPP/98fvvtN9fnHTt28MEHH/DHH39w4YUXMn78eC677DI6dOjAhRdeSKdOnc74rq1bt1JQUMCAAQNOeb1z587UqlXLda5Pnz44HA62b99OTEwMw4cP57zzzuPQoUM0bNiQefPmcfnllxMREQHAn3/+yV9//cW8efNcZRiGgcPhICEhgbZt2wK4tdgUWbBgAS+//DK7d+8mOzubwsJCwsLCXNe3b9/OmDFj3J7p0aMHP/74I+Bsgdm9eze33HILt912m+uewsJCwsPD3Z4LDg4mNzf3jHUm4m1KMESkXOrUqYPFYiE9Pf20991xxx08//zzOBwO/vjjD/79738TEhJC3759WblyZakSjODgYI/jPffcc2nevDnz58/nrrvu4vPPP2fu3Lmu69nZ2dxxxx2MGzeu2LNNmzZ1/fnEJAZgzZo1DB8+nEcffZRBgwYRHh7O/Pnzef7550sdW9FYkbfffpuePXu6XTu5SyktLY169eqVumwRb1GCISLlEhAQQLt27diyZcsp18GYM2cOUVFRDB482JWIFA1+tNls2O32Ur2rZcuWBAcHs2zZMm699dZi19u2bcvcuXPJyclxJQC//PILPj4+tG7d2nXf8OHDmTdvHo0bN8bHx4fLL7/cde2cc85hy5YttGjRonQV8I/Vq1cTGxvLww8/7Dq3d+9et3tat27Nb7/9xogRI1znTmzliYmJoWHDhuzZs6fEacYn2rRpU7EZRCKVkQZ5iki5DRo0iFWrVpV4LSUlhccff5xXXnkFgMjISNq2bcvs2bNZs2YNy5Ytc01vPZOgoCAmTZrExIkT+eCDD9i9eze//vqraz2I4cOHExQUxMiRI9m0aRPLly/nnnvu4aabbnINBC26b8OGDcyaNYvrrruOwMBA17VJkyaxevVqxo4dy8aNG9m5cydffvllsUGeJ2vZsiX79u1j/vz57N69m5dffrnYLI977rmHOXPm8P7777Nz504ef/xx/vrrLywWi+ueRx99lCeffJKXX36ZHTt28Pfff/Pee+/xwgsvuO7Jzc1l/fr1WthMqgQlGCJSbrfccguLFy8mIyOj2LXx48dz//3307BhQ9e5uXPnMn/+fK644goefPBBzj33XMC5DojFYmHFihWnfNfUqVO5//77mTZtGm3btmXo0KGkpKQAEBISwnfffUdaWhrnnnsu1113HQMGDODVV191K6NFixb06NGDv/76q1hLQadOnVi5ciU7duzgggsuoGvXrkybNs0t/pIMHjyY++67j7Fjx9KlSxdWr17tml1SZPjw4UyZMoUHHniAc845h4SEBEaNGuU2fuXWW2/lnXfe4b333qNjx4707duXuXPn0qxZM9c9X375JU2bNuWCCy44bUwilYG6SESk3Nq1a8fll1/O66+/zpQpU9yuffLJJ8Xu79Gjh2tdjBMlJCQQERFB586dT/kuHx8fHn74YbeuiBN17NjRNWjydNauXXvKa+eeey7ff//9Ka8nJiaWeP6ZZ57hmWeecTt37733un2eOnWqW+Jx8cUXF+uOueGGG7jhhhtO+f6XXnqJadOmnfK6SGWiBENEPPLss8/y9ddfe1TG4sWLeeihh4iMjDQpqsolNzeXN998k0GDBuHr68snn3zCDz/8wNKlS0tdRmpqKtdccw3/+c9/KjBSEfMowRA5i8LDw1m0aBGLFi0qdm3QoEEARERElDgVEpy/xTdu3JgHHnigxOsPPfRQiefbtm17yjKDg4OJjo7miSeeKNalABRbPvxkcXFx3HPPPae950yeffZZj56v7CwWC4sXL2bWrFnk5+fTunVrPvvsMwYOHFjqMurWrcvEiRMrMEoRc2mzMxERETGdBnmKiIiI6ZRgiIiIiOmUYIiIiIjplGCIiIiI6ZRgiIiIiOmUYIiIiIjplGCIiIiI6ZRgiIiIiOn+H3duKrImS3KEAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "KWJjF_haSsmb"
      },
      "execution_count": 13,
      "outputs": []
    }
  ]
}