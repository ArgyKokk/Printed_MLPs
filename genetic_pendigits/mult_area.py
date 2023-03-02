#mult_area[i][j] -> multiplier area with input i-bit and weight j
#mult_area[1][j] is not zero, but I assumed it to be zero because it is only an AND gate.
#example: from mult_area import mult_area; print(mult_area[3][12])

mult_area=[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.31726e+00, 2.33670e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.33670e+00, 2.33670e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 2.69534e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.31726e+00, 2.33670e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.61368e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 2.61368e+00, 2.31726e+00, 2.31726e+00, 2.69534e+00, 2.69534e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.69534e+00, 2.33670e+00, 2.33670e+00, 2.31726e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.31726e+00, 2.33670e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.33670e+00, 2.33670e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.31726e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.61368e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.61368e+00, 2.33670e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.53473e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 2.61368e+00, 2.31726e+00, 2.31726e+00, 2.61368e+00, 2.33670e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.69534e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.69534e+00, 2.69534e+00, 2.33670e+00, 2.33670e+00, 2.33670e+00, 2.69534e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 2.31726e+00, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 4.89090e+00, 0.00000e+00, 3.96184e+00, 4.89090e+00, 4.71979e+00, 0.00000e+00, 0.00000e+00, 3.96184e+00, 8.28622e+00, 4.89090e+00, 8.63806e+00, 4.71979e+00, 5.06585e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.29966e+00, 3.96184e+00, 4.78779e+00, 8.28622e+00, 8.50008e+00, 4.89090e+00, 4.89090e+00, 8.63806e+00, 9.85600e+00, 4.71979e+00, 7.20618e+00, 5.06585e+00, 4.95309e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.89090e+00, 0.00000e+00, 5.20185e+00, 7.29966e+00, 8.90536e+00, 3.96184e+00, 3.96184e+00, 4.78779e+00, 8.28622e+00, 8.28622e+00, 1.00921e+01, 8.50008e+00, 8.19082e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 6.18156e+00, 8.63806e+00, 1.01058e+01, 9.85600e+00, 5.31198e+00, 4.71979e+00, 5.34302e+00, 7.20618e+00, 9.34846e+00, 5.06585e+00, 8.00518e+00, 4.95309e+00, 4.95309e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.89090e+00, 0.00000e+00, 3.96184e+00, 4.89090e+00, 5.53354e+00, 0.00000e+00, 0.00000e+00, 5.20185e+00, 9.69858e+00, 7.29966e+00, 1.02747e+01, 8.90536e+00, 9.19408e+00, 3.96184e+00, 3.96184e+00, 3.96184e+00, 1.09261e+01, 4.78779e+00, 4.78779e+00, 8.28622e+00, 1.06015e+01, 8.28622e+00, 8.28622e+00, 1.00921e+01, 1.05401e+01, 8.50008e+00, 7.98980e+00, 8.19082e+00, 8.19082e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 9.01524e+00, 6.18156e+00, 8.95302e+00, 8.63806e+00, 8.63806e+00, 1.01058e+01, 1.07677e+01, 9.85600e+00, 9.07836e+00, 5.31198e+00, 5.38171e+00, 4.71979e+00, 5.34302e+00, 5.34302e+00, 1.00608e+01, 7.20618e+00, 9.63630e+00, 9.34846e+00, 5.38171e+00, 5.06585e+00, 5.55688e+00, 8.00518e+00, 1.00008e+01, 4.95309e+00, 7.63008e+00, 4.95309e+00, 4.95309e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.89090e+00, 0.00000e+00, 3.96184e+00, 4.89090e+00, 5.53354e+00, 0.00000e+00, 0.00000e+00, 3.96184e+00, 8.28622e+00, 4.89090e+00, 8.63806e+00, 5.53354e+00, 5.55688e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.71372e+00, 5.20185e+00, 5.44971e+00, 9.69858e+00, 1.02642e+01, 7.29966e+00, 7.29966e+00, 1.10746e+01, 1.23997e+01, 9.29222e+00, 1.01398e+01, 1.04013e+01, 1.04013e+01, 4.50810e+00, 4.50810e+00, 4.50810e+00, 8.10546e+00, 4.50810e+00, 5.44971e+00, 9.85016e+00, 9.72572e+00, 4.78779e+00, 4.78779e+00, 4.78779e+00, 8.28622e+00, 8.28622e+00, 1.05684e+01, 1.13811e+01, 1.06764e+01, 9.10060e+00, 9.10060e+00, 9.10060e+00, 9.72572e+00, 1.00581e+01, 1.16833e+01, 1.12158e+01, 8.57494e+00, 8.50008e+00, 8.02380e+00, 8.35422e+00, 1.18639e+01, 8.19082e+00, 8.30566e+00, 8.19082e+00, 8.19082e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 4.89090e+00, 8.61574e+00, 4.89090e+00, 8.15402e+00, 4.89090e+00, 4.89090e+00, 9.82880e+00, 9.72572e+00, 6.18156e+00, 9.30678e+00, 9.24944e+00, 9.46330e+00, 6.68630e+00, 6.68630e+00, 6.68630e+00, 1.19982e+01, 9.03838e+00, 9.03838e+00, 1.07677e+01, 1.06958e+01, 9.85600e+00, 9.25822e+00, 9.01614e+00, 1.09106e+01, 5.31198e+00, 8.27166e+00, 5.38171e+00, 5.38171e+00, 4.58976e+00, 5.34302e+00, 5.34302e+00, 9.63630e+00, 5.34302e+00, 1.04537e+01, 1.00608e+01, 1.02020e+01, 7.20618e+00, 7.39182e+00, 1.13120e+01, 1.10836e+01, 9.34846e+00, 1.03885e+01, 5.38171e+00, 5.38171e+00, 5.06585e+00, 5.55688e+00, 5.55688e+00, 9.77897e+00, 8.12002e+00, 1.02582e+01, 1.01192e+01, 5.38171e+00, 5.06585e+00, 5.57632e+00, 6.96816e+00, 1.01192e+01, 5.08529e+00, 6.96816e+00, 5.06585e+00, 5.06585e+00, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 7.31976e+00, 0.00000e+00, 8.15210e+00, 7.31976e+00, 8.93232e+00, 0.00000e+00, 5.77243e+00, 8.15210e+00, 1.39784e+01, 7.31976e+00, 1.50824e+01, 8.93232e+00, 6.99446e+00, 0.00000e+00, 0.00000e+00, 5.77243e+00, 1.60236e+01, 8.15210e+00, 1.10793e+01, 1.39784e+01, 1.56558e+01, 7.31976e+00, 1.17219e+01, 1.50824e+01, 1.96321e+01, 8.93232e+00, 1.39588e+01, 6.99446e+00, 7.33366e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.32722e+01, 5.77243e+00, 1.49915e+01, 1.60236e+01, 1.86261e+01, 8.15210e+00, 1.18813e+01, 1.10793e+01, 1.84888e+01, 1.39784e+01, 1.85367e+01, 1.56558e+01, 1.30477e+01, 7.31976e+00, 7.31976e+00, 1.17219e+01, 2.01634e+01, 1.50824e+01, 2.00984e+01, 1.96321e+01, 1.84033e+01, 8.93232e+00, 8.88717e+00, 1.39588e+01, 2.20832e+01, 6.99446e+00, 1.39887e+01, 7.33366e+00, 7.33366e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.31976e+00, 0.00000e+00, 1.32618e+01, 1.32722e+01, 1.45339e+01, 5.77243e+00, 5.83465e+00, 1.49915e+01, 2.40872e+01, 1.60236e+01, 2.65278e+01, 1.77123e+01, 1.81028e+01, 8.15210e+00, 8.35702e+00, 1.18813e+01, 2.08366e+01, 1.10793e+01, 1.10793e+01, 1.73789e+01, 2.27987e+01, 1.43578e+01, 1.90992e+01, 1.88885e+01, 2.55038e+01, 1.51932e+01, 2.06645e+01, 1.30477e+01, 1.41530e+01, 9.13932e+00, 9.13932e+00, 9.13932e+00, 1.60638e+01, 1.35066e+01, 2.13473e+01, 2.01634e+01, 2.31912e+01, 1.46683e+01, 1.88113e+01, 1.91039e+01, 2.63560e+01, 1.96321e+01, 2.25955e+01, 1.69640e+01, 1.16822e+01, 8.93232e+00, 8.93232e+00, 8.17544e+00, 2.10410e+01, 1.39588e+01, 1.97153e+01, 2.20832e+01, 1.74691e+01, 6.99446e+00, 9.35252e+00, 1.39887e+01, 2.07979e+01, 7.29966e+00, 1.33676e+01, 7.29966e+00, 7.29966e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.31976e+00, 0.00000e+00, 8.35702e+00, 7.31976e+00, 8.93232e+00, 0.00000e+00, 6.86785e+00, 1.32618e+01, 1.80051e+01, 1.32722e+01, 2.11389e+01, 1.49733e+01, 1.39120e+01, 5.14090e+00, 5.14090e+00, 5.83465e+00, 2.00284e+01, 1.49915e+01, 1.85530e+01, 2.35146e+01, 2.32428e+01, 1.60236e+01, 1.87883e+01, 2.65278e+01, 2.61005e+01, 2.07159e+01, 2.12329e+01, 1.81028e+01, 1.83584e+01, 8.15210e+00, 8.35702e+00, 8.35702e+00, 2.23324e+01, 1.14672e+01, 1.53114e+01, 2.08366e+01, 2.34362e+01, 1.10793e+01, 1.64537e+01, 1.10793e+01, 1.89924e+01, 1.73789e+01, 2.40553e+01, 2.26744e+01, 1.96317e+01, 1.51354e+01, 1.51354e+01, 1.91800e+01, 2.62059e+01, 1.88885e+01, 2.25576e+01, 2.55038e+01, 1.92793e+01, 1.51932e+01, 1.69077e+01, 2.06645e+01, 2.92556e+01, 1.16238e+01, 1.62446e+01, 1.39059e+01, 1.39059e+01, 9.13932e+00, 9.13932e+00, 9.13932e+00, 9.13932e+00, 9.13932e+00, 1.93231e+01, 1.64779e+01, 2.07131e+01, 1.26122e+01, 1.66601e+01, 2.13473e+01, 2.69651e+01, 1.93158e+01, 2.40054e+01, 2.31912e+01, 2.27781e+01, 1.54667e+01, 1.54667e+01, 1.79832e+01, 2.66366e+01, 2.02840e+01, 2.22698e+01, 2.71842e+01, 2.51800e+01, 1.93697e+01, 2.02609e+01, 2.28028e+01, 2.43313e+01, 1.68453e+01, 2.13904e+01, 1.21360e+01, 1.18920e+01, 8.93232e+00, 8.93232e+00, 8.93232e+00, 1.79221e+01, 8.88717e+00, 1.92463e+01, 2.04304e+01, 2.39798e+01, 1.39588e+01, 1.75371e+01, 2.05883e+01, 2.97255e+01, 2.20832e+01, 2.73277e+01, 1.84179e+01, 8.74944e+00, 6.99446e+00, 6.99446e+00, 9.35252e+00, 2.21008e+01, 1.37408e+01, 1.81962e+01, 2.03193e+01, 1.78661e+01, 7.33366e+00, 1.02125e+01, 1.26435e+01, 1.92198e+01, 7.33366e+00, 1.26435e+01, 7.33366e+00, 7.33366e+00, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 1.15682e+01, 0.00000e+00, 1.22869e+01, 1.15682e+01, 1.21574e+01, 0.00000e+00, 9.63246e+00, 1.22869e+01, 2.54005e+01, 1.15682e+01, 2.62419e+01, 1.21574e+01, 1.16629e+01, 0.00000e+00, 7.82041e+00, 9.63246e+00, 2.48887e+01, 1.22869e+01, 2.30471e+01, 2.54005e+01, 2.54574e+01, 1.15682e+01, 2.19068e+01, 2.62419e+01, 3.03052e+01, 1.21574e+01, 1.95310e+01, 1.16629e+01, 1.07782e+01, 0.00000e+00, 0.00000e+00, 7.82041e+00, 2.09500e+01, 9.63246e+00, 2.19483e+01, 2.53028e+01, 2.83307e+01, 1.22869e+01, 2.06285e+01, 2.30471e+01, 3.84474e+01, 2.46372e+01, 3.61593e+01, 2.54574e+01, 2.09652e+01, 1.33877e+01, 1.60839e+01, 2.19068e+01, 3.55668e+01, 2.46195e+01, 3.63800e+01, 3.10003e+01, 2.95720e+01, 1.23616e+01, 1.46720e+01, 1.92346e+01, 3.28892e+01, 1.10677e+01, 1.87942e+01, 1.07782e+01, 1.06084e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.99391e+01, 7.18888e+00, 2.14331e+01, 2.09500e+01, 2.56395e+01, 9.63246e+00, 1.29683e+01, 2.19483e+01, 3.58865e+01, 2.53028e+01, 3.84712e+01, 2.83307e+01, 2.61703e+01, 1.22869e+01, 1.77027e+01, 2.06285e+01, 3.27407e+01, 2.30471e+01, 2.60743e+01, 3.84474e+01, 3.68263e+01, 2.46372e+01, 3.32101e+01, 3.58837e+01, 4.85446e+01, 2.54574e+01, 3.22340e+01, 2.09652e+01, 2.04696e+01, 1.33877e+01, 1.33877e+01, 1.60839e+01, 2.66770e+01, 2.19068e+01, 3.62572e+01, 3.60443e+01, 3.89268e+01, 2.47857e+01, 3.48264e+01, 3.57651e+01, 4.72203e+01, 3.03052e+01, 3.30778e+01, 2.99773e+01, 2.59234e+01, 1.23616e+01, 1.56072e+01, 1.46720e+01, 3.27002e+01, 1.92346e+01, 3.28546e+01, 3.28892e+01, 3.03205e+01, 1.28873e+01, 1.66500e+01, 1.87942e+01, 2.65367e+01, 1.07782e+01, 1.53822e+01, 1.09386e+01, 1.09386e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.15682e+01, 0.00000e+00, 1.85612e+01, 1.99391e+01, 2.21026e+01, 7.82041e+00, 1.89712e+01, 2.14331e+01, 3.55626e+01, 2.09500e+01, 3.54973e+01, 2.56395e+01, 2.11406e+01, 9.63246e+00, 1.74839e+01, 1.29683e+01, 3.30389e+01, 2.19483e+01, 3.53185e+01, 3.58865e+01, 4.28460e+01, 2.53028e+01, 3.36945e+01, 3.84712e+01, 4.09719e+01, 2.83307e+01, 3.29616e+01, 2.61703e+01, 2.45233e+01, 1.22869e+01, 1.29418e+01, 1.77027e+01, 3.34905e+01, 2.06285e+01, 2.67880e+01, 3.32696e+01, 4.39899e+01, 2.30471e+01, 3.10827e+01, 2.61421e+01, 3.34925e+01, 3.74948e+01, 4.62327e+01, 3.57017e+01, 3.65228e+01, 2.46372e+01, 3.04759e+01, 3.32101e+01, 5.11723e+01, 3.61593e+01, 4.59705e+01, 4.85446e+01, 4.11769e+01, 2.54574e+01, 2.85967e+01, 3.34446e+01, 4.32593e+01, 2.13792e+01, 2.97294e+01, 2.09459e+01, 2.04938e+01, 1.33877e+01, 1.33877e+01, 1.33877e+01, 2.05905e+01, 1.60839e+01, 3.25253e+01, 2.66770e+01, 3.61774e+01, 2.19068e+01, 2.91554e+01, 3.62572e+01, 5.36184e+01, 3.60443e+01, 4.33881e+01, 3.90971e+01, 3.95903e+01, 2.42054e+01, 2.86919e+01, 3.48264e+01, 5.13086e+01, 3.69089e+01, 4.12625e+01, 4.75505e+01, 3.84133e+01, 3.07193e+01, 3.21259e+01, 3.29261e+01, 4.54893e+01, 2.99773e+01, 3.82664e+01, 2.59234e+01, 1.70506e+01, 1.31467e+01, 1.31467e+01, 1.56072e+01, 3.04978e+01, 1.46720e+01, 3.27977e+01, 3.27002e+01, 3.10471e+01, 1.95310e+01, 2.65954e+01, 3.24406e+01, 4.33748e+01, 3.28892e+01, 3.61317e+01, 3.03205e+01, 2.91288e+01, 1.16629e+01, 1.25783e+01, 1.66500e+01, 3.04334e+01, 1.78553e+01, 2.87079e+01, 2.91468e+01, 2.79528e+01, 1.07782e+01, 1.41112e+01, 1.53822e+01, 3.05568e+01, 1.09386e+01, 1.80887e+01, 1.09386e+01, 1.09386e+01, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 1.59746e+01, 0.00000e+00, 1.61739e+01, 1.59746e+01, 1.68530e+01, 0.00000e+00, 1.35194e+01, 1.61739e+01, 3.37021e+01, 1.59746e+01, 3.17634e+01, 1.68530e+01, 1.36612e+01, 0.00000e+00, 1.16804e+01, 1.35194e+01, 3.19377e+01, 1.61739e+01, 3.39167e+01, 3.31721e+01, 3.27358e+01, 1.63123e+01, 2.99625e+01, 3.26080e+01, 3.84283e+01, 1.68695e+01, 2.57361e+01, 1.48326e+01, 1.30526e+01, 0.00000e+00, 8.42138e+00, 1.16804e+01, 2.91814e+01, 1.35194e+01, 3.29902e+01, 3.19377e+01, 3.45504e+01, 1.61739e+01, 3.18770e+01, 3.39167e+01, 5.20743e+01, 3.31721e+01, 5.25363e+01, 3.27358e+01, 3.38580e+01, 1.63123e+01, 2.66696e+01, 2.99625e+01, 4.49514e+01, 3.30221e+01, 5.54906e+01, 3.73192e+01, 3.32366e+01, 1.68695e+01, 2.07510e+01, 2.50679e+01, 3.58347e+01, 1.48326e+01, 2.46738e+01, 1.30526e+01, 1.37137e+01, 0.00000e+00, 0.00000e+00, 8.63885e+00, 2.72699e+01, 1.16804e+01, 2.77960e+01, 2.91814e+01, 3.22072e+01, 1.35194e+01, 1.90889e+01, 3.25762e+01, 5.33383e+01, 3.23518e+01, 4.67313e+01, 3.45504e+01, 3.55626e+01, 1.61739e+01, 2.45155e+01, 3.10489e+01, 4.57996e+01, 3.32937e+01, 3.93632e+01, 5.14746e+01, 4.86058e+01, 3.31721e+01, 4.53272e+01, 5.23648e+01, 5.99496e+01, 3.27358e+01, 4.25272e+01, 3.38580e+01, 2.81900e+01, 1.63123e+01, 1.99709e+01, 2.70836e+01, 4.83879e+01, 2.99625e+01, 5.03446e+01, 4.53655e+01, 4.98512e+01, 3.30221e+01, 4.62048e+01, 5.54906e+01, 6.04287e+01, 3.64910e+01, 4.66767e+01, 3.32366e+01, 3.59522e+01, 1.68695e+01, 2.26667e+01, 2.07510e+01, 3.99929e+01, 2.58592e+01, 4.04722e+01, 3.54207e+01, 3.43022e+01, 1.48326e+01, 2.26618e+01, 2.46738e+01, 3.67221e+01, 1.30526e+01, 2.44191e+01, 1.37137e+01, 1.43025e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.28480e+01, 8.63885e+00, 2.75219e+01, 2.72699e+01, 3.13452e+01, 1.16804e+01, 2.54613e+01, 2.77960e+01, 5.10839e+01, 2.91814e+01, 4.87487e+01, 3.22072e+01, 3.11362e+01, 1.35194e+01, 2.40558e+01, 1.90889e+01, 4.58948e+01, 3.22769e+01, 4.62719e+01, 5.35667e+01, 5.52339e+01, 3.19377e+01, 4.45577e+01, 4.75594e+01, 5.64626e+01, 3.41363e+01, 4.15230e+01, 3.51485e+01, 3.46528e+01, 1.61739e+01, 2.15897e+01, 2.45155e+01, 4.77285e+01, 3.06348e+01, 4.38128e+01, 4.66277e+01, 5.42377e+01, 3.08911e+01, 4.48050e+01, 3.93632e+01, 5.60920e+01, 5.16408e+01, 7.11013e+01, 4.77777e+01, 5.22254e+01, 3.31721e+01, 4.16017e+01, 4.49132e+01, 6.88999e+01, 5.15367e+01, 7.02451e+01, 6.11918e+01, 5.59465e+01, 3.27358e+01, 3.59588e+01, 4.21905e+01, 5.69543e+01, 3.46106e+01, 4.22737e+01, 2.79083e+01, 2.81431e+01, 1.63123e+01, 1.63123e+01, 1.99709e+01, 3.17216e+01, 2.70836e+01, 4.46108e+01, 4.85431e+01, 5.13939e+01, 2.97963e+01, 4.05640e+01, 4.93364e+01, 6.21431e+01, 4.62614e+01, 6.99665e+01, 4.95562e+01, 5.13343e+01, 3.30221e+01, 4.13957e+01, 4.62048e+01, 6.87504e+01, 5.53063e+01, 6.76695e+01, 6.07735e+01, 5.70233e+01, 3.73192e+01, 4.21430e+01, 4.65743e+01, 5.91351e+01, 3.32366e+01, 3.97647e+01, 3.59522e+01, 3.02236e+01, 1.68530e+01, 2.05653e+01, 2.26667e+01, 4.14889e+01, 2.07510e+01, 3.76723e+01, 4.22528e+01, 4.95834e+01, 2.58592e+01, 3.64265e+01, 4.08862e+01, 6.20233e+01, 3.53514e+01, 4.45233e+01, 3.47162e+01, 3.53256e+01, 1.62602e+01, 1.79925e+01, 2.26618e+01, 4.05003e+01, 2.46738e+01, 3.65012e+01, 3.67221e+01, 3.63504e+01, 1.30526e+01, 2.32179e+01, 2.47564e+01, 3.65073e+01, 1.37137e+01, 2.27084e+01, 1.35413e+01, 1.35413e+01, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 1.87039e+01, 0.00000e+00, 1.96468e+01, 1.87039e+01, 1.88393e+01, 0.00000e+00, 1.74064e+01, 1.96468e+01, 4.13921e+01, 2.01993e+01, 4.08444e+01, 1.93231e+01, 1.79593e+01, 0.00000e+00, 1.55674e+01, 1.74064e+01, 3.97466e+01, 1.96468e+01, 3.91260e+01, 4.13921e+01, 4.00293e+01, 2.01993e+01, 3.90270e+01, 4.04303e+01, 4.52771e+01, 2.00640e+01, 3.27553e+01, 1.79593e+01, 1.89305e+01, 0.00000e+00, 1.29129e+01, 1.55674e+01, 3.76569e+01, 1.74064e+01, 4.18177e+01, 3.97466e+01, 4.18438e+01, 1.96468e+01, 3.80122e+01, 3.93311e+01, 6.44616e+01, 4.18061e+01, 6.80250e+01, 4.00293e+01, 4.29623e+01, 2.01993e+01, 3.55534e+01, 3.94411e+01, 5.81507e+01, 4.00163e+01, 7.07700e+01, 4.48631e+01, 3.79991e+01, 1.93231e+01, 2.89138e+01, 3.47156e+01, 4.38203e+01, 1.79593e+01, 3.02108e+01, 1.89305e+01, 1.84717e+01, 0.00000e+00, 1.06868e+01, 1.29129e+01, 3.56454e+01, 1.55674e+01, 3.66652e+01, 3.76569e+01, 3.72579e+01, 1.74064e+01, 3.33205e+01, 4.18177e+01, 6.67375e+01, 3.97466e+01, 6.69095e+01, 4.18438e+01, 4.09567e+01, 1.96468e+01, 3.61932e+01, 3.54130e+01, 6.40822e+01, 3.91260e+01, 6.05561e+01, 6.40476e+01, 7.06281e+01, 4.13921e+01, 5.68820e+01, 6.69937e+01, 7.02512e+01, 3.96152e+01, 5.75336e+01, 4.31962e+01, 3.94974e+01, 2.01993e+01, 3.27936e+01, 3.51393e+01, 5.75925e+01, 3.86129e+01, 6.38290e+01, 5.80106e+01, 6.36136e+01, 4.16725e+01, 6.29392e+01, 6.99419e+01, 7.24656e+01, 4.55194e+01, 5.38078e+01, 3.79991e+01, 4.20606e+01, 1.88393e+01, 2.86531e+01, 2.89138e+01, 5.14166e+01, 3.69216e+01, 5.42648e+01, 4.50819e+01, 4.20170e+01, 1.79593e+01, 2.97767e+01, 3.12217e+01, 4.29915e+01, 1.89305e+01, 2.99206e+01, 1.84717e+01, 1.63205e+01, 0.00000e+00, 0.00000e+00, 1.06868e+01, 3.20044e+01, 1.29129e+01, 3.09324e+01, 3.56454e+01, 3.89267e+01, 1.55674e+01, 3.00267e+01, 3.76081e+01, 5.82713e+01, 3.76569e+01, 6.30143e+01, 3.76720e+01, 4.05788e+01, 1.74064e+01, 2.88941e+01, 3.29064e+01, 5.65160e+01, 4.18177e+01, 6.06851e+01, 6.56615e+01, 6.67276e+01, 3.97466e+01, 5.51166e+01, 6.46535e+01, 6.76609e+01, 4.18438e+01, 5.41301e+01, 4.09567e+01, 3.98583e+01, 1.96468e+01, 2.84025e+01, 3.58940e+01, 5.92730e+01, 3.54130e+01, 5.56465e+01, 6.43701e+01, 7.02993e+01, 3.87120e+01, 5.82688e+01, 5.89634e+01, 8.83291e+01, 6.44616e+01, 8.92077e+01, 7.16226e+01, 6.06256e+01, 4.18061e+01, 5.49962e+01, 5.89379e+01, 8.22665e+01, 6.61656e+01, 8.84254e+01, 7.16596e+01, 6.30526e+01, 3.92012e+01, 4.87628e+01, 5.75336e+01, 7.10074e+01, 4.22186e+01, 5.72865e+01, 4.11174e+01, 3.45447e+01, 2.01993e+01, 2.38579e+01, 3.27936e+01, 5.81552e+01, 3.49731e+01, 6.01583e+01, 5.68538e+01, 5.93537e+01, 3.77848e+01, 5.31928e+01, 6.28708e+01, 8.51751e+01, 5.85543e+01, 7.98604e+01, 6.31995e+01, 6.44998e+01, 3.94196e+01, 5.48388e+01, 6.33490e+01, 8.55745e+01, 7.04543e+01, 8.60424e+01, 7.32980e+01, 7.12301e+01, 4.52966e+01, 4.80790e+01, 5.33937e+01, 7.32656e+01, 3.79991e+01, 5.00089e+01, 4.32959e+01, 4.02115e+01, 1.88393e+01, 2.75480e+01, 2.82390e+01, 4.91435e+01, 2.89138e+01, 4.95981e+01, 5.11355e+01, 6.98113e+01, 3.45424e+01, 4.79279e+01, 5.20151e+01, 6.90109e+01, 4.34062e+01, 5.40249e+01, 4.16029e+01, 4.17452e+01, 1.75288e+01, 2.35357e+01, 2.93626e+01, 5.18818e+01, 3.12985e+01, 4.96538e+01, 4.29915e+01, 4.31271e+01, 1.78990e+01, 2.91148e+01, 2.99206e+01, 4.62222e+01, 1.92333e+01, 2.82858e+01, 1.63205e+01, 1.67093e+01, 0.00000e+00],
[0, 0.00000e+00, 0.00000e+00, 2.25909e+01, 0.00000e+00, 2.39479e+01, 2.40863e+01, 2.33534e+01, 0.00000e+00, 2.04653e+01, 2.39479e+01, 4.92852e+01, 2.40863e+01, 4.82527e+01, 2.36802e+01, 2.36685e+01, 0.00000e+00, 1.90403e+01, 2.04653e+01, 4.70400e+01, 2.39479e+01, 4.73391e+01, 4.92852e+01, 4.69086e+01, 2.40863e+01, 4.71347e+01, 4.86667e+01, 5.70880e+01, 2.33534e+01, 4.00418e+01, 2.33417e+01, 2.22234e+01, 0.00000e+00, 1.67999e+01, 1.90403e+01, 4.48466e+01, 2.04653e+01, 5.05554e+01, 4.70400e+01, 4.93873e+01, 2.39479e+01, 4.52156e+01, 4.69483e+01, 8.58504e+01, 4.92852e+01, 7.75875e+01, 4.77368e+01, 5.16501e+01, 2.40863e+01, 4.42038e+01, 4.71347e+01, 7.33636e+01, 4.94948e+01, 7.68201e+01, 5.63026e+01, 4.66562e+01, 2.29569e+01, 3.45559e+01, 4.29992e+01, 5.27051e+01, 2.36685e+01, 3.82617e+01, 2.22234e+01, 2.09703e+01, 0.00000e+00, 1.49609e+01, 1.67999e+01, 4.47468e+01, 1.90403e+01, 4.55177e+01, 4.48466e+01, 4.51474e+01, 2.04653e+01, 4.46954e+01, 4.97273e+01, 7.54796e+01, 4.70400e+01, 7.94698e+01, 5.02154e+01, 4.88838e+01, 2.39479e+01, 4.46181e+01, 4.48015e+01, 7.81816e+01, 4.65343e+01, 7.31218e+01, 8.50222e+01, 8.33833e+01, 4.88711e+01, 7.56197e+01, 7.78159e+01, 7.99097e+01, 4.64946e+01, 7.00521e+01, 5.06588e+01, 4.64212e+01, 2.40863e+01, 4.07302e+01, 4.46179e+01, 6.46796e+01, 4.79628e+01, 7.93796e+01, 7.29495e+01, 7.39441e+01, 4.78222e+01, 7.46587e+01, 7.86553e+01, 8.68348e+01, 5.68151e+01, 7.92898e+01, 4.66562e+01, 4.83501e+01, 2.29569e+01, 3.42599e+01, 3.52324e+01, 7.40598e+01, 4.14585e+01, 7.29970e+01, 5.31192e+01, 5.15969e+01, 2.24098e+01, 3.58592e+01, 3.87054e+01, 5.21754e+01, 2.22234e+01, 3.59629e+01, 2.09703e+01, 1.88427e+01, 0.00000e+00, 1.19193e+01, 1.49609e+01, 3.98218e+01, 1.67999e+01, 4.35431e+01, 4.43327e+01, 4.55548e+01, 1.90403e+01, 4.25949e+01, 4.55177e+01, 7.48439e+01, 4.48466e+01, 7.46139e+01, 4.50450e+01, 4.50430e+01, 2.12934e+01, 4.25173e+01, 4.32956e+01, 7.27899e+01, 5.01055e+01, 7.87489e+01, 7.71358e+01, 8.58318e+01, 4.70400e+01, 7.20330e+01, 7.86417e+01, 7.88580e+01, 5.09562e+01, 7.78546e+01, 4.84698e+01, 4.94495e+01, 2.39479e+01, 4.25147e+01, 4.46181e+01, 7.54291e+01, 4.52156e+01, 7.29019e+01, 7.56545e+01, 8.09130e+01, 4.73624e+01, 7.88804e+01, 7.42339e+01, 1.01845e+02, 8.00390e+01, 1.04545e+02, 8.36292e+01, 8.00609e+01, 4.92852e+01, 7.58213e+01, 7.72430e+01, 1.00746e+02, 7.74684e+01, 1.08463e+02, 7.99097e+01, 7.25811e+01, 4.74211e+01, 6.45823e+01, 7.11510e+01, 7.81081e+01, 5.08691e+01, 6.65400e+01, 4.64212e+01, 4.98188e+01, 2.40863e+01, 3.74373e+01, 3.97359e+01, 6.61944e+01, 4.46179e+01, 7.37540e+01, 6.48436e+01, 6.44608e+01, 4.57098e+01, 7.51779e+01, 7.76491e+01, 1.03793e+02, 7.39742e+01, 1.03397e+02, 7.42502e+01, 7.27771e+01, 4.94784e+01, 8.16859e+01, 7.59173e+01, 1.03337e+02, 7.87881e+01, 1.06297e+02, 8.55927e+01, 8.60377e+01, 5.58458e+01, 6.32518e+01, 7.70701e+01, 8.58356e+01, 4.63405e+01, 6.33619e+01, 4.94482e+01, 4.64690e+01, 2.43188e+01, 3.25206e+01, 3.42599e+01, 6.65564e+01, 3.53688e+01, 6.30776e+01, 6.98125e+01, 7.91816e+01, 4.08056e+01, 5.92742e+01, 7.14227e+01, 8.29251e+01, 5.13937e+01, 6.36489e+01, 5.11828e+01, 4.54699e+01, 2.20388e+01, 2.94998e+01, 3.90033e+01, 6.21132e+01, 3.80974e+01, 6.15413e+01, 5.05191e+01, 4.97266e+01, 2.25696e+01, 3.52159e+01, 3.59629e+01, 4.73426e+01, 2.12222e+01, 3.66769e+01, 1.88427e+01, 1.92560e+01, 0.00000e+00]
]
