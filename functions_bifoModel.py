import numpy as np
from scipy import optimize as opt


def uniFlowQ(w, s, d, g, c):
    # Uniform flow discharge computation
    # Wide rectangular channel assumption is made
    return w * c * (g * s) ** 0.5 * d**1.5


def uniFlowD(q, w, s, g, c):
    # Uniform water depth computation
    return (q / (w * c * np.sqrt(g * s))) ** (2 / 3)


def uniFlowS(q, w, d, g, c):
    # Uniform/gradually varying flow bed/energy slope computation
    # (wide rectangular cross-section hypothesis is made)
    return (q / (w * c * g**0.5 * d**1.5)) ** 2


def phis_scalar(theta, tf, c):
    phi = 0
    phiD = 0
    phiT = 0
    if tf == "MPM":
        theta_cr = 0.047
        if theta > theta_cr:
            phi = 8 * (theta - theta_cr) ** 1.5
            phiT = 1.5 * theta / (theta - theta_cr)
    elif tf == "EH":
        cD = 2.5 / c
        phi = 0.05 * c**2 * theta**2.5
        phiD = 2 * cD
        # phiD = 0
        phiT = 2.5
    elif tf == "P90":  # Parker (1990)
        A = 0.0386
        B = 0.853
        C = 5474
        D = 0.00218
        x = theta / A
        if x > 1.59:
            phi = C * D * theta**1.5 * (1 - B / x) ** 4.5
            Phi_der = 1.5 * theta**0.5 * (1 - B / x) ** 4.5 * C * D + 4.5 * A * B * (
                1.0 - B / x
            ) ** 3.5 * C * D / (theta**0.5)
        elif x >= 1:
            phi = D * theta**1.5 * (np.exp(14.2 * (x - 1) - 9.28 * (x - 1) ** 2))
            Phi_der = (
                1.0 / A * phi * (14.2 - 9.28 * 2.0 * (x - 1.0)) + 1.5 * phi / theta
            )
        else:
            phi = D * theta**1.5 * x**14.2
            Phi_der = (
                14.2 / A * D * theta**1.5 * x**13.2
                + D * x**14.2 * 1.5 * theta**0.5
            )
        phiT = theta / phi * Phi_der
    elif tf == "P78":  # Parker (1978)
        theta_cr = 0.03
        if theta > theta_cr:
            phi = 11.2 * theta**1.5 * (1 - theta_cr / theta) ** 4.5
            phiT = 1.5 + 4.5 * theta_cr / (theta - theta_cr)
    elif tf == "WP":  # Wong&Parker (2006)
        theta_cr = 0.0495
        phi = 3.97 * (theta - theta_cr) ** 1.5
        phiT = 1.5 * theta / (theta - theta_cr)
    else:
        print("error: unknown transport formula")
        phi = None
    return phi, phiD, phiT


def phis(theta, tf, c):
    # Computes phi and phiT given array values of theta and D
    phi = np.zeros(np.size(theta))
    phiD = np.zeros(np.size(theta))
    phiT = np.zeros(np.size(theta))
    if tf == "MPM":  # Meyer-Peter and Muller (1948)
        theta_cr = 0.047
        nst = theta < theta_cr
        phi[~nst] = 8 * (theta[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * theta[~nst] / (theta[~nst] - theta_cr)
    elif tf == "EH":  # Engelund & Hansen
        phi = 0.05 * c**2 * theta**2.5
        phiD = 2 * 2.5 / c
        phiT = 2.5
    elif tf == "P90":  # Parker (1990)
        a = 0.0386
        b = 0.853
        c = 5474
        d = 0.00218
        x = theta / a
        phi[x >= 1] = (
            d
            * (theta[x >= 1] ** 1.5)
            * (np.exp(14.2 * (x[x >= 1] - 1) - 9.28 * (x[x >= 1] - 1) ** 2))
        )
        phiT[x >= 1] = (
            theta[x >= 1]
            / phi[x >= 1]
            * (
                1 / a * phi[x >= 1] * (14.2 - 9.28 * 2 * (x[x >= 1] - 1))
                + 1.5 * phi[x >= 1] / theta[x >= 1]
            )
        )
        phi[x > 1.59] = c * d * theta[x > 1.59] ** 1.5 * (1 - b / x[x > 1.59]) ** 4.5
        phiT[x > 1.59] = (
            theta[x > 1.59]
            / phi[x > 1.59]
            * (
                1.5 * theta[x > 1.59] ** 0.5 * (1 - b / x[x > 1.59]) ** 4.5 * c * d
                + 4.5
                * a
                * b
                * (1 - b / x[x > 1.59]) ** 3.5
                * c
                * d
                / theta[x > 1.59] ** 0.5
            )
        )
        phi[x < 1] = d * theta[x < 1] ** 1.5 * x[x < 1] ** 14.2
        phiT[x < 1] = (
            theta[x < 1]
            / phi[x < 1]
            * (
                14.2 / a * d * theta[x < 1] ** 1.5 * x[x < 1] ** 13.2
                + d * x[x < 1] ** 14.2 * 1.5 * theta[x < 1] ** 0.5
            )
        )
    elif tf == "P78":  # Parker (1978)
        theta_cr = 0.03
        nst = theta < theta_cr
        phi[~nst] = 11.2 * theta[~nst] ** 1.5 * (1 - theta_cr / theta[~nst]) ** 4.5
        phiT[~nst] = 1.5 + 4.5 * theta_cr / (theta[~nst] - theta_cr)
        phiT[nst] = None
    elif tf == "WP":  # Wong&Parker (2006)
        theta_cr = 0.0495
        nst = theta < theta_cr
        phi[~nst] = 3.97 * (theta[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * theta[~nst] / (theta[~nst] - theta_cr)
    else:
        print("error: unknown transport formula")
        phi = None
        phiT = None
    return phi, phiD, phiT


def betaR_MR(theta, r, phiD, phiT, c0, cD=-1):
    if cD == -1:
        cD = 2.5 / c0
    betaR = (
        np.pi
        / (2 * np.sqrt(2))
        * c0
        * np.sqrt(r)
        / (theta**0.25 * np.sqrt(phiD + phiT - (1.5 + cD)))
    )
    return betaR


def betaC_MR(theta, alpha, r, phiD, phiT, c0, cD=-1):
    if cD == -1:
        cD = 2.5 / c0
    betaC = 4 * alpha * r / (theta**0.5) / (phiT + phiD - (1.5 + cD))
    return betaC


def fSys_BRT(x,tf,theta0,Q0,Qs0,D0,W0,S0,W_b,W_c,alpha,r,g,delta,d50,C0):
    res     = np.zeros((3,))
    D_b,D_c = x[:2]*D0
    S_b     = x[2]*S0
    S_c     = x[2]*S0
    theta_b = S_b*D_b/(delta*d50)
    theta_c = S_c*D_c/(delta*d50)
    Qs_b    = W_b*np.sqrt(g*delta*d50**3)*phis_scalar(theta_b,tf,C0)[0]
    Qs_c    = W_c*np.sqrt(g*delta*d50**3)*phis_scalar(theta_c,tf,C0)[0]
    Q_b     = uniFlowQ(W_b,S_b,D_b,g,C0)
    Q_c     = uniFlowQ(W_c,S_c,D_c,g,C0)
    Qs_y    = 0.5*(Qs_b-Qs_c)
    Q_y     = 0.5*(Q_b-Q_c)
    res [0] = Qs_y/Qs0-Q_y/Q0+2*alpha*r/np.sqrt(theta0)*(D_c-D_b)/(0.5*(W0+W_b+W_c))
    res [1] = (Qs_b+Qs_c)/Qs0-1
    res [2] = (Q_b+Q_c)/Q0-1
    return res


def shieldsUpdate(q, w, d, d50, g, delta, c):
    j = uniFlowS(q, w, d, g, c)
    theta = j * d / (delta * d50)
    return theta


def C_eta(Q, W, D, g, delta, d50, p, C0, tf, eps=1e-6):
    """
    Computes the celerity of the propagation of a perturbation of the bed elvel
    (kinematic wave approximation)
    """
    theta_epsp = shieldsUpdate(Q + eps, W, D, d50, g, delta, C0)
    theta_epsm = shieldsUpdate(Q - eps, W, D, d50, g, delta, C0)
    Qs_epsp = W * np.sqrt(g * delta * d50**3) * phis(theta_epsp, tf, C0)[0]
    Qs_epsm = W * np.sqrt(g * delta * d50**3) * phis(theta_epsm, tf, C0)[0]
    dQsdQ = (Qs_epsp - Qs_epsm) / (2 * eps)
    Fr = Q / (W * D * np.sqrt(g * D))
    C_eta = dQsdQ * (Q / (W * D)) / (1 - Fr**2) / (1 - p)
    return C_eta


def profilesF(s, q, w, d, g, c):
    j = uniFlowS(q, w, d, g, c)
    Fr = q / (w * d * np.sqrt(g * d))
    return (s - j) / (1 - Fr**2)


def buildProfile_rk4(dd, q, w, s, dx, g, c):
    """
    Integrates the gradually-varied flow equation in upstream direction
    using the Runge-Kutta 4 method
    """
    d = np.zeros(np.size(s) + 1)
    d[-1] = dd
    for i in range(np.size(s), 0, -1):
        k1 = profilesF(s[i - 1], q, w, d[i], g, c)
        k2 = profilesF(s[i - 1], q, w, d[i] - dx / 2 * k1, g, c)
        k3 = profilesF(s[i - 1], q, w, d[i] - dx / 2 * k2, g, c)
        k4 = profilesF(s[i - 1], q, w, d[i] - dx * k3, g, c)
        d[i - 1] = d[i] - dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return d


def fSys_rk4(Q_b, Ddb, Ddc, D_0, inStep, Q_0, W_b, W_c, S_b, S_c, dx, g, c):
    """
    fSys = 0 is solved to solve the flow partition problem at the node for each time step
    """
    Q_c = Q_0 - Q_b
    D_b = buildProfile_rk4(Ddb, Q_b, W_b, S_b, dx, g, c)
    D_c = buildProfile_rk4(Ddc, Q_c, W_c, S_c, dx, g, c)
    return (D_b[0] - D_c[0] + inStep * D_0) / ((D_b[0] + D_c[0]) / 2)


def triStep_F(D_avg, deltaEta, S_ab, S_ac, Q, W, C, g):
    # Returns dD/dx over a triangular step, given a the average water depth
    # and the bed elevation gap deltaEta between the two halves of the cross-section
    #! Here deltaEta is dimensional (contrary to inlet step in bifoModel)
    D_ab = D_avg - deltaEta / 2
    D_ac = D_avg + deltaEta / 2
    S = 0.5 * (S_ab + S_ac)
    j = (Q / (C * g**0.5 * W * 0.5 * (D_ab**1.5 + D_ac**1.5))) ** 2
    Fr = Q / (W * D_avg * np.sqrt(g * D_avg))
    return (S - j) / (1 - Fr**2)


def buildTriStepProfile(DD, eta_a, eta_b, eta_c, alpha, W_a, Q, C, g, M=10):
    """
    Compute the water-surface profiles over a triangular step, given the bed elevations
    at the ends of the node cells and the average water depth at the downstream end
    of the node cells
    """
    #! Here deltaEta (dEta) is dimensional (contrary to the inlet step)
    dx = alpha * W_a / (M - 1)
    D_avg = np.zeros(M)
    D_avg[-1] = DD
    S_ab = (eta_a - eta_b) / (alpha * W_a)
    S_ac = (eta_a - eta_c) / (alpha * W_a)
    dEta = np.linspace(0, eta_b - eta_c, M)
    for i in range(M - 1, 0, -1):
        k1 = triStep_F(D_avg[i], dEta[i], S_ab, S_ac, Q, W_a, C, g)
        k2 = triStep_F(
            D_avg[i] - dx / 2 * k1,
            0.5 * (dEta[i] + dEta[i - 1]),
            S_ab,
            S_ac,
            Q,
            W_a,
            C,
            g,
        )
        k3 = triStep_F(
            D_avg[i] - dx / 2 * k2,
            0.5 * (dEta[i] + dEta[i - 1]),
            S_ab,
            S_ac,
            Q,
            W_a,
            C,
            g,
        )
        k4 = triStep_F(D_avg[i] - dx * k3, dEta[i - 1], S_ab, S_ac, Q, W_a, C, g)
        D_avg[i - 1] = D_avg[i] - dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return D_avg
