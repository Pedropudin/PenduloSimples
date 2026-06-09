import numpy as np


def criar_malha(T, m):
    """
    Cria a malha temporal no intervalo [0, T].

    Entrada:
        T: tempo final
        m: número de pontos internos

    Saída:
        h: passo da malha
        t: vetor completo com os contornos
        t_interno: vetor apenas com os pontos internos
    """
    h = T / (m + 1)
    t = np.linspace(0.0, T, m + 2)
    t_interno = t[1:-1]

    return h, t, t_interno


def calcular_G(theta, h, alpha, beta, g, L):
    """
    Calcula o vetor residual G(theta) do sistema não linear.

    Para cada ponto interno:
        G_i(theta) =
        (theta_{i-1} - 2 theta_i + theta_{i+1}) / h^2
        + (g/L) sin(theta_i)
    """
    theta = np.asarray(theta, dtype=float)
    m = theta.size

    G = np.zeros(m, dtype=float)

    for i in range(m):
        # Nos extremos internos, usamos as condições de contorno.
        theta_esq = alpha if i == 0 else theta[i - 1]
        theta_dir = beta if i == m - 1 else theta[i + 1]

        G[i] = (
            (theta_esq - 2.0 * theta[i] + theta_dir) / h**2
            + (g / L) * np.sin(theta[i])
        )

    return G


def montar_diagonais_jacobiana(theta, h, g, L):
    """
    Monta as três diagonais da Jacobiana tridiagonal.

    A Jacobiana tem:
        diagonal inferior:  1/h^2
        diagonal principal: -2/h^2 + (g/L) cos(theta_i)
        diagonal superior:  1/h^2
    """
    theta = np.asarray(theta, dtype=float)
    m = theta.size

    lower = np.full(m - 1, 1.0 / h**2, dtype=float)
    diag = -2.0 / h**2 + (g / L) * np.cos(theta)
    upper = np.full(m - 1, 1.0 / h**2, dtype=float)

    return lower, diag, upper


def calcular_jacobiana(theta, h, g, L):
    """
    Monta a Jacobiana completa.

    Esta função é útil para conferência e depuração.
    O método principal usa apenas as diagonais.
    """
    lower, diag, upper = montar_diagonais_jacobiana(theta, h, g, L)
    m = diag.size

    J = np.zeros((m, m), dtype=float)

    for i in range(m):
        J[i, i] = diag[i]

        if i > 0:
            J[i, i - 1] = lower[i - 1]

        if i < m - 1:
            J[i, i + 1] = upper[i]

    return J


def resolver_tridiagonal_thomas(lower, diag, upper, b):
    """
    Resolve Ax = b pelo algoritmo de Thomas.

    A matriz A é tridiagonal e é representada por:
        lower: diagonal inferior
        diag: diagonal principal
        upper: diagonal superior

    Este método evita usar np.linalg.solve no núcleo do projeto.
    """
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    b = np.asarray(b, dtype=float)

    n = diag.size

    # Vetores modificados durante a eliminação.
    c_mod = np.zeros(n - 1, dtype=float)
    d_mod = np.zeros(n, dtype=float)

    # Primeira linha.
    if diag[0] == 0.0:
        raise ZeroDivisionError("Pivô nulo no algoritmo de Thomas.")

    if n > 1:
        c_mod[0] = upper[0] / diag[0]

    d_mod[0] = b[0] / diag[0]

    # Eliminação progressiva.
    for i in range(1, n):
        denominador = diag[i] - lower[i - 1] * c_mod[i - 1]

        if denominador == 0.0:
            raise ZeroDivisionError("Pivô nulo no algoritmo de Thomas.")

        if i < n - 1:
            c_mod[i] = upper[i] / denominador

        d_mod[i] = (b[i] - lower[i - 1] * d_mod[i - 1]) / denominador

    # Substituição regressiva.
    x = np.zeros(n, dtype=float)
    x[-1] = d_mod[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_mod[i] - c_mod[i] * x[i + 1]

    return x


def newton_raphson_pendulo(
    theta0,
    h,
    alpha,
    beta,
    g,
    L,
    tol=1e-10,
    max_iter=50
):
    """
    Resolve G(theta) = 0 pelo método de Newton-Raphson.

    Em cada iteração:
        J(theta_k) delta_k = -G(theta_k)
        theta_{k+1} = theta_k + delta_k
    """
    theta_atual = np.asarray(theta0, dtype=float).copy()

    erros = []
    residuos = []

    for k in range(max_iter):
        G = calcular_G(theta_atual, h, alpha, beta, g, L)
        lower, diag, upper = montar_diagonais_jacobiana(theta_atual, h, g, L)

        # Resolve J delta = -G usando o solver tridiagonal próprio.
        delta = resolver_tridiagonal_thomas(lower, diag, upper, -G)

        theta_novo = theta_atual + delta

        erro_relativo = np.linalg.norm(theta_novo - theta_atual, ord=np.inf)
        erro_relativo /= np.linalg.norm(theta_novo, ord=np.inf)

        residuo = np.linalg.norm(
            calcular_G(theta_novo, h, alpha, beta, g, L),
            ord=np.inf
        )

        erros.append(erro_relativo)
        residuos.append(residuo)

        if erro_relativo < tol:
            return theta_novo, erros, residuos, k + 1

        theta_atual = theta_novo

    return theta_atual, erros, residuos, max_iter


def montar_solucao_com_contorno(theta_interno, alpha, beta):
    """
    Junta os valores de contorno com a solução interna.

    Resultado:
        [alpha, theta_1, theta_2, ..., theta_m, beta]
    """
    theta_interno = np.asarray(theta_interno, dtype=float)

    return np.concatenate(([alpha], theta_interno, [beta]))


def resolver_pendulo_linearizado(m, h, alpha, beta, g, L):
    """
    Resolve o modelo linearizado:
        theta'' + (g/L) theta = 0

    Esse modelo vem da aproximação:
        sin(theta) ≈ theta
    """
    lower = np.full(m - 1, 1.0 / h**2, dtype=float)
    diag = np.full(m, -2.0 / h**2 + g / L, dtype=float)
    upper = np.full(m - 1, 1.0 / h**2, dtype=float)

    b = np.zeros(m, dtype=float)

    # Contribuições das condições de contorno.
    b[0] -= alpha / h**2
    b[-1] -= beta / h**2

    return resolver_tridiagonal_thomas(lower, diag, upper, b)


def resolver_pendulo_nao_linear(
    T,
    m,
    alpha,
    beta,
    g,
    L,
    tipo_chute="seno",
    tol=1e-10,
    max_iter=50
):
    """
    Resolve o PVC não linear do pêndulo para parâmetros gerais.

    Esta função é usada principalmente nas comparações:
        - Terra vs Mercúrio
        - sensibilidade em relação a L
        - sensibilidade em relação a alpha e beta
        - sensibilidade em relação a g
    """
    h, t, t_interno = criar_malha(T, m)

    if tipo_chute == "constante":
        theta0 = np.full(m, alpha, dtype=float)
    else:
        theta0 = alpha - np.sin(t_interno / 2.0)

    theta_interno, erros, residuos, iteracoes = newton_raphson_pendulo(
        theta0=theta0,
        h=h,
        alpha=alpha,
        beta=beta,
        g=g,
        L=L,
        tol=tol,
        max_iter=max_iter
    )

    theta_completo = montar_solucao_com_contorno(theta_interno, alpha, beta)

    return t, theta_completo, erros, residuos, iteracoes