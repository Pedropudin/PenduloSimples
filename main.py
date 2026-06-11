import os
import numpy as np

from pendulo import (
    criar_malha,
    calcular_G,
    newton_raphson_pendulo,
    montar_solucao_com_contorno,
    resolver_pendulo_linearizado,
    resolver_pendulo_nao_linear,
)

from visualizacao import (
    salvar_grafico_linhas,
    salvar_grafico_convergencia,
    salvar_grafico_sensibilidade,
    fechar_graficos,
)

from simulacao import animar_dois_pendulos


def imprimir_resultado(nome, iteracoes, erros, residuos, theta):
    """
    Imprime um resumo da solução obtida pelo método de Newton-Raphson.
    """
    print(f"\n=== {nome} ===")
    print(f"Iterações: {iteracoes}")
    print(f"Último erro relativo: {erros[-1]:.3e}")
    print(f"Último resíduo: {residuos[-1]:.3e}")
    print("Primeiros valores da solução:")
    print(theta[:10])


def estudar_sensibilidade(
    nome,
    valores,
    resolver_caso,
    titulo_grafico,
    arquivo_saida,
    tol,
    max_iter
):
    """
    Executa um estudo de sensibilidade variando um parâmetro.

    Os casos que não convergem são mostrados no terminal,
    mas não são colocados no gráfico.
    """
    curvas = []

    print(f"\n=== Sensibilidade em relação a {nome} ===")

    for valor in valores:
        t, theta, erros, residuos, iteracoes, label = resolver_caso(valor)

        print(
            f"{label} | "
            f"Iterações = {iteracoes} | "
            f"Erro final = {erros[-1]:.3e} | "
            f"Resíduo final = {residuos[-1]:.3e}"
        )

        convergiu = erros[-1] < tol and iteracoes < max_iter

        if not convergiu:
            print("  -> Não convergiu. Não será plotado.")
            continue

        curvas.append((t, theta, label))

    salvar_grafico_sensibilidade(
        curvas=curvas,
        titulo=titulo_grafico,
        nome_arquivo=arquivo_saida
    )


def main():
    # Pasta onde os gráficos serão salvos
    os.makedirs("resultados", exist_ok=True)

    # Configuração apenas para imprimir vetores de forma mais legível
    np.set_printoptions(precision=8, suppress=True)

    # Parâmetros oficiais do problema
    T = 2 * np.pi
    alpha = 0.7
    beta = 0.7
    g = 9.8
    L = 1.0
    m = 100

    # Parâmetros numéricos
    tol = 1e-10
    max_iter = 50

    # Malha temporal
    h, t, t_interno = criar_malha(T, m)

    # Chute inicial (a): constante 0.7
    theta0_a = np.full(m, 0.7, dtype=float)

    # Chute inicial (b): theta_i = 0.7 - sin(t_i/2)
    theta0_b = 0.7 - np.sin(t_interno / 2)

    # Resolve o PVC não linear com os dois chutes iniciais
    theta_a, erros_a, residuos_a, iter_a = newton_raphson_pendulo(
        theta0_a, h, alpha, beta, g, L, tol, max_iter
    )

    theta_b, erros_b, residuos_b, iter_b = newton_raphson_pendulo(
        theta0_b, h, alpha, beta, g, L, tol, max_iter
    )

    # Adiciona os valores de contorno alpha e beta à solução interna
    theta_completo_a = montar_solucao_com_contorno(theta_a, alpha, beta)
    theta_completo_b = montar_solucao_com_contorno(theta_b, alpha, beta)

    # Resultados básicos
    imprimir_resultado(
        "Resultado com chute inicial (a): constante 0.7",
        iter_a,
        erros_a,
        residuos_a,
        theta_completo_a
    )

    imprimir_resultado(
        "Resultado com chute inicial (b): 0.7 - sin(t_i/2)",
        iter_b,
        erros_b,
        residuos_b,
        theta_completo_b
    )

    diferenca = np.linalg.norm(theta_completo_a - theta_completo_b, ord=np.inf)

    print("\n=== Comparação entre as duas soluções ===")
    print(f"Diferença máxima entre as soluções: {diferenca:.3e}")

    residuo_a = np.linalg.norm(
        calcular_G(theta_a, h, alpha, beta, g, L),
        ord=np.inf
    )

    residuo_b = np.linalg.norm(
        calcular_G(theta_b, h, alpha, beta, g, L),
        ord=np.inf
    )

    print("\n=== Norma do resíduo ||G(theta)||_inf ===")
    print(f"Resíduo do chute (a): {residuo_a:.3e}")
    print(f"Resíduo do chute (b): {residuo_b:.3e}")

    # Gráfico do erro relativo em escala logarítmica
    salvar_grafico_convergencia(
        erros_a=erros_a,
        erros_b=erros_b,
        nome_arquivo="resultados/erro_convergencia.png"
    )

    # Gráfico comparando as duas soluções não lineares
    salvar_grafico_linhas(
        dados=[
            (t, theta_completo_a, "Solução com chute (a)"),
            (t, theta_completo_b, "Solução com chute (b)")
        ],
        titulo="Comparação das soluções obtidas",
        nome_arquivo="resultados/comparacao_solucoes.png"
    )

    # Comparação com o modelo linearizado sin(theta) ≈ theta
    theta_linear = resolver_pendulo_linearizado(
        m=m,
        h=h,
        alpha=alpha,
        beta=beta,
        g=g,
        L=L
    )

    theta_linear_completo = montar_solucao_com_contorno(
        theta_linear,
        alpha,
        beta
    )

    dif_a_linear = np.linalg.norm(
        theta_completo_a - theta_linear_completo,
        ord=np.inf
    )

    dif_b_linear = np.linalg.norm(
        theta_completo_b - theta_linear_completo,
        ord=np.inf
    )

    print("\n=== Comparação com o modelo linearizado ===")
    print(f"Diferença máxima entre solução não linear (a) e linearizada: {dif_a_linear:.3e}")
    print(f"Diferença máxima entre solução não linear (b) e linearizada: {dif_b_linear:.3e}")

    salvar_grafico_linhas(
        dados=[
            (t, theta_completo_a, "Não linear - chute (a)"),
            (t, theta_completo_b, "Não linear - chute (b)"),
            (t, theta_linear_completo, "Linearizado")
        ],
        titulo="Comparação entre modelo não linear e modelo linearizado",
        nome_arquivo="resultados/comparacao_linearizado.png"
    )

    # Comparação Terra vs Mercúrio
    g_terra = 9.8
    g_mercurio = 3.7

    t_terra, theta_terra, erros_terra, residuos_terra, iter_terra = resolver_pendulo_nao_linear(
        T=T,
        m=m,
        alpha=alpha,
        beta=beta,
        g=g_terra,
        L=L,
        tipo_chute="seno",
        tol=tol,
        max_iter=max_iter
    )

    t_mercurio, theta_mercurio, erros_mercurio, residuos_mercurio, iter_mercurio = resolver_pendulo_nao_linear(
        T=T,
        m=m,
        alpha=alpha,
        beta=beta,
        g=g_mercurio,
        L=L,
        tipo_chute="seno",
        tol=tol,
        max_iter=max_iter
    )

    dif_terra_mercurio = np.linalg.norm(
        theta_terra - theta_mercurio,
        ord=np.inf
    )

    print("\n=== Comparação Terra vs Mercúrio ===")
    print(f"Iterações na Terra: {iter_terra}")
    print(f"Erro final na Terra: {erros_terra[-1]:.3e}")
    print(f"Resíduo final na Terra: {residuos_terra[-1]:.3e}")
    print(f"Iterações em Mercúrio: {iter_mercurio}")
    print(f"Erro final em Mercúrio: {erros_mercurio[-1]:.3e}")
    print(f"Resíduo final em Mercúrio: {residuos_mercurio[-1]:.3e}")
    print(f"Diferença máxima entre Terra e Mercúrio: {dif_terra_mercurio:.3e}")

    salvar_grafico_linhas(
        dados=[
            (t_terra, theta_terra, "Terra: g = 9.8"),
            (t_mercurio, theta_mercurio, "Mercúrio: g = 3.7")
        ],
        titulo="Comparação da solução: Terra vs Mercúrio",
        nome_arquivo="resultados/comparacao_terra_mercurio.png"
    )

    # Sensibilidade em relação ao comprimento L
    def caso_L(L_teste):
        t_L, theta_L, erros_L, residuos_L, iter_L = resolver_pendulo_nao_linear(
            T=T,
            m=m,
            alpha=alpha,
            beta=beta,
            g=g,
            L=L_teste,
            tipo_chute="seno",
            tol=tol,
            max_iter=max_iter
        )

        return (
            t_L,
            theta_L,
            erros_L,
            residuos_L,
            iter_L,
            f"L = {L_teste:.2f}"
        )

    estudar_sensibilidade(
        nome="L",
        valores=[0.8, 1.0, 1.2, 1.5, 2.0],
        resolver_caso=caso_L,
        titulo_grafico="Sensibilidade em relação ao comprimento L",
        arquivo_saida="resultados/sensibilidade_L.png",
        tol=tol,
        max_iter=max_iter
    )

    # Sensibilidade em relação às condições de contorno alpha = beta
    def caso_alpha_beta(valor):
        t_ab, theta_ab, erros_ab, residuos_ab, iter_ab = resolver_pendulo_nao_linear(
            T=T,
            m=m,
            alpha=valor,
            beta=valor,
            g=g,
            L=L,
            tipo_chute="seno",
            tol=tol,
            max_iter=max_iter
        )

        return (
            t_ab,
            theta_ab,
            erros_ab,
            residuos_ab,
            iter_ab,
            f"alpha = beta = {valor:.2f}"
        )

    estudar_sensibilidade(
        nome="alpha = beta",
        valores=[0.3, 0.7, 1.0],
        resolver_caso=caso_alpha_beta,
        titulo_grafico="Sensibilidade em relação às condições de contorno",
        arquivo_saida="resultados/sensibilidade_alpha_beta.png",
        tol=tol,
        max_iter=max_iter
    )

    # Sensibilidade em relação à gravidade g
    def caso_g(g_teste):
        t_g, theta_g, erros_g, residuos_g, iter_g = resolver_pendulo_nao_linear(
            T=T,
            m=m,
            alpha=alpha,
            beta=beta,
            g=g_teste,
            L=L,
            tipo_chute="seno",
            tol=tol,
            max_iter=max_iter
        )

        return (
            t_g,
            theta_g,
            erros_g,
            residuos_g,
            iter_g,
            f"g = {g_teste:.2f}"
        )

    estudar_sensibilidade(
        nome="g",
        valores=[3.7, 6.0, 9.8, 12.0],
        resolver_caso=caso_g,
        titulo_grafico="Sensibilidade em relação à gravidade g",
        arquivo_saida="resultados/sensibilidade_g.png",
        tol=tol,
        max_iter=max_iter
    )

    # Animação lado a lado: Terra vs Mercúrio
    executar_simulacao = True

    if executar_simulacao:
        animar_dois_pendulos(
            theta_esq=theta_terra,
            theta_dir=theta_mercurio,
            L_esq=L,
            L_dir=L,
            T=T,
            titulo_esq="Terra",
            titulo_dir="Mercúrio",
            g_esq=g_terra,
            g_dir=g_mercurio,
            titulo_janela="Pêndulo simples: Terra vs Mercúrio",
            #salvar_arquivo="resultados/simulacao_terra_mercurio.gif"
        )

    fechar_graficos()


if __name__ == "__main__":
    main()