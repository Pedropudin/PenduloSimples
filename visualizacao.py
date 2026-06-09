import matplotlib.pyplot as plt

def salvar_grafico_linhas(
    dados,
    titulo,
    nome_arquivo,
    xlabel="t",
    ylabel="theta(t)"
):
    """
    Salva um gráfico com uma ou mais curvas.

    Entrada:
        dados: lista de tuplas no formato (x, y, label)
        titulo: título do gráfico
        nome_arquivo: caminho onde o gráfico será salvo
        xlabel: nome do eixo x
        ylabel: nome do eixo y
    """
    plt.figure()

    for x, y, label in dados:
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")


def salvar_grafico_convergencia(erros_a, erros_b, nome_arquivo):
    """
    Salva o gráfico de convergência do método de Newton-Raphson.

    O eixo y usa escala logarítmica, como pedido no enunciado.
    """
    plt.figure()

    plt.semilogy(
        range(1, len(erros_a) + 1),
        erros_a,
        marker="o",
        label="Chute (a): constante 0.7"
    )

    plt.semilogy(
        range(1, len(erros_b) + 1),
        erros_b,
        marker="o",
        label="Chute (b): 0.7 - sin(t_i/2)"
    )

    plt.xlabel("Iteração")
    plt.ylabel("Erro relativo")
    plt.title("Convergência do método de Newton-Raphson")
    plt.grid(True)
    plt.legend()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")


def salvar_grafico_sensibilidade(curvas, titulo, nome_arquivo):
    """
    Salva um gráfico de sensibilidade.

    Entrada:
        curvas: lista de tuplas no formato (t, theta, label)
    """
    salvar_grafico_linhas(
        dados=curvas,
        titulo=titulo,
        nome_arquivo=nome_arquivo,
        xlabel="t",
        ylabel="theta(t)"
    )


def fechar_graficos():
    """
    Fecha todas as figuras abertas pelo Matplotlib.
    """
    plt.close("all")