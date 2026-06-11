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

    # Animação lado a lado: Terra vs Mercúrio
    executar_simulacao = True

    if executar_simulacao:
        animar_dois_pendulos(
            theta_esq=theta_completo_b,
            theta_dir=theta_linear_completo,
            L_esq=L,
            L_dir=L,
            T=T,
            titulo_esq="Não Linearizado",
            titulo_dir="Linearizado",
            g_esq=g,
            g_dir=g,
            titulo_janela="Pêndulo simples: Comparação de Linearização",
            #salvar_arquivo="resultados/simulacao_terra_mercurio.gif"
        )

    fechar_graficos()


if __name__ == "__main__":
    main()
