import math
import os

import pygame
import numpy as np
from PIL import Image


def animar_dois_pendulos(
    theta_esq,
    theta_dir,
    L_esq,
    L_dir,
    T,
    titulo_esq="Terra",
    titulo_dir="Mercúrio",
    g_esq=None,
    g_dir=None,
    titulo_janela="Comparação de pêndulos",
    salvar_arquivo=None,
    max_frames_salvos=200
):
    """
    Anima dois pêndulos lado a lado usando soluções numéricas já calculadas.

    Parâmetros:
        theta_esq:
            Vetor completo de ângulos do pêndulo da esquerda.
            Deve incluir os contornos:
            [alpha, theta_1, ..., theta_m, beta]

        theta_dir:
            Vetor completo de ângulos do pêndulo da direita.

        L_esq:
            Comprimento do pêndulo da esquerda.

        L_dir:
            Comprimento do pêndulo da direita.

        T:
            Tempo final do intervalo [0, T].

        titulo_esq:
            Nome exibido acima do pêndulo da esquerda.

        titulo_dir:
            Nome exibido acima do pêndulo da direita.

        g_esq, g_dir:
            Valores opcionais de gravidade para mostrar na tela.

        titulo_janela:
            Título da janela do Pygame.

        salvar_arquivo:
            Caminho opcional para salvar a animação como GIF.
            Exemplo: "resultados/simulacao_terra_mercurio.gif".

        max_frames_salvos:
            Número máximo aproximado de frames salvos no GIF.
            Esse limite evita arquivos muito grandes quando a malha tem muitos pontos.
    """

    theta_esq = np.asarray(theta_esq, dtype=float)
    theta_dir = np.asarray(theta_dir, dtype=float)

    if theta_esq.ndim != 1 or theta_dir.ndim != 1:
        raise ValueError("theta_esq e theta_dir devem ser vetores unidimensionais.")

    if len(theta_esq) < 2 or len(theta_dir) < 2:
        raise ValueError("Os vetores theta precisam ter pelo menos dois pontos.")

    if len(theta_esq) != len(theta_dir):
        raise ValueError("theta_esq e theta_dir devem ter o mesmo tamanho para comparação lado a lado.")

    if L_esq <= 0 or L_dir <= 0:
        raise ValueError("Os comprimentos L_esq e L_dir devem ser positivos.")

    if T <= 0:
        raise ValueError("T deve ser positivo.")

    if max_frames_salvos <= 0:
        raise ValueError("max_frames_salvos deve ser positivo.")

    # Inicialização do Pygame
    pygame.init()

    largura = 1100
    altura = 650
    tela = pygame.display.set_mode((largura, altura))
    pygame.display.set_caption(titulo_janela)

    clock = pygame.time.Clock()

    # Cores
    branco = (255, 255, 255)
    preto = (30, 30, 30)
    azul = (30, 120, 220)
    vermelho = (220, 80, 60)
    cinza = (180, 180, 180)
    cinza_claro = (225, 225, 225)

    fonte_titulo = pygame.font.SysFont("Arial", 30, bold=True)
    fonte_info = pygame.font.SysFont("Arial", 22)
    fonte_pequena = pygame.font.SysFont("Arial", 18)

    # Pivôs dos dois pêndulos
    pivo_esq = (largura // 4, 130)
    pivo_dir = (3 * largura // 4, 130)

    # Escala visual.
    # Mantemos a mesma escala para os dois para comparação justa.
    escala_base = 300

    maior_L = max(L_esq, L_dir)
    tamanho_maximo = altura - 230

    if maior_L * escala_base > tamanho_maximo:
        escala = tamanho_maximo / maior_L
    else:
        escala = escala_base

    # Passo temporal da solução discreta
    numero_intervalos = len(theta_esq) - 1
    h = T / numero_intervalos

    # FPS visual. Pode ajustar para 20, 30, 40 se quiser.
    fps = 20

    # Preparação opcional para salvar GIF.
    # Para evitar arquivos gigantes, salvamos no máximo max_frames_salvos
    # ao longo de uma volta completa da solução discreta.
    frames_salvos = []
    indices_salvos = set()
    passo_gif = max(1, math.ceil(len(theta_esq) / max_frames_salvos))
    indices_para_salvar = set(range(0, len(theta_esq), passo_gif))

    frame = 0
    rodando = True
    pausado = False

    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False

                if evento.key == pygame.K_SPACE:
                    pausado = not pausado

                # R reinicia a animação
                if evento.key == pygame.K_r:
                    frame = 0

        tela.fill(branco)

        # Linha vertical separando os cenários
        pygame.draw.line(
            tela,
            cinza_claro,
            (largura // 2, 0),
            (largura // 2, altura),
            2
        )

        indice = frame % len(theta_esq)
        tempo_atual = indice * h

        angulo_esq = theta_esq[indice]
        angulo_dir = theta_dir[indice]

        # Desenha os dois pêndulos
        _desenhar_pendulo(
            tela=tela,
            pivo=pivo_esq,
            theta=angulo_esq,
            L=L_esq,
            escala=escala,
            cor_massa=azul,
            cor_haste=preto,
            cor_referencia=cinza
        )

        _desenhar_pendulo(
            tela=tela,
            pivo=pivo_dir,
            theta=angulo_dir,
            L=L_dir,
            escala=escala,
            cor_massa=vermelho,
            cor_haste=preto,
            cor_referencia=cinza
        )

        # Títulos
        texto_titulo_esq = fonte_titulo.render(titulo_esq, True, preto)
        texto_titulo_dir = fonte_titulo.render(titulo_dir, True, preto)

        tela.blit(
            texto_titulo_esq,
            (pivo_esq[0] - texto_titulo_esq.get_width() // 2, 35)
        )

        tela.blit(
            texto_titulo_dir,
            (pivo_dir[0] - texto_titulo_dir.get_width() // 2, 35)
        )

        # Informações de gravidade, se fornecidas
        if g_esq is not None:
            texto_g_esq = fonte_info.render(f"g = {g_esq:.2f} m/s²", True, preto)
            tela.blit(
                texto_g_esq,
                (pivo_esq[0] - texto_g_esq.get_width() // 2, 75)
            )

        if g_dir is not None:
            texto_g_dir = fonte_info.render(f"g = {g_dir:.2f} m/s²", True, preto)
            tela.blit(
                texto_g_dir,
                (pivo_dir[0] - texto_g_dir.get_width() // 2, 75)
            )

        # Informações no rodapé
        texto_tempo = fonte_info.render(
            f"Tempo: {tempo_atual:.2f} s / {T:.2f} s",
            True,
            preto
        )

        texto_angulo_esq = fonte_pequena.render(
            f"{titulo_esq}: theta = {angulo_esq:.4f} rad",
            True,
            preto
        )

        texto_angulo_dir = fonte_pequena.render(
            f"{titulo_dir}: theta = {angulo_dir:.4f} rad",
            True,
            preto
        )

        texto_controles = fonte_pequena.render(
            "Espaço: pausar | R: reiniciar | ESC: sair",
            True,
            preto
        )

        tela.blit(texto_tempo, (20, altura - 90))
        tela.blit(texto_angulo_esq, (20, altura - 65))
        tela.blit(texto_angulo_dir, (20, altura - 40))
        tela.blit(texto_controles, (largura - texto_controles.get_width() - 20, altura - 40))

        pygame.display.flip()

        # Captura opcional do frame para o GIF.
        # Capturamos cada índice selecionado uma única vez.
        if (
            salvar_arquivo is not None
            and indice in indices_para_salvar
            and indice not in indices_salvos
        ):
            imagem = pygame.surfarray.array3d(tela)
            imagem = np.transpose(imagem, (1, 0, 2))
            frames_salvos.append(Image.fromarray(imagem))
            indices_salvos.add(indice)

        if not pausado:
            frame += 1

        clock.tick(fps)

    if salvar_arquivo is not None and frames_salvos:
        pasta_saida = os.path.dirname(salvar_arquivo)
        if pasta_saida:
            os.makedirs(pasta_saida, exist_ok=True)

        duracao_ms = max(1, int(1000 * passo_gif / fps))

        frames_salvos[0].save(
            salvar_arquivo,
            save_all=True,
            append_images=frames_salvos[1:],
            duration=duracao_ms,
            loop=0
        )

        print(f"Animação salva em: {salvar_arquivo}")

    pygame.quit()


def _desenhar_pendulo(
    tela,
    pivo,
    theta,
    L,
    escala,
    cor_massa,
    cor_haste,
    cor_referencia
):
    """
    Função auxiliar para desenhar um único pêndulo.

    A conversão usada é:
        x = x_pivo + L * escala * sin(theta)
        y = y_pivo + L * escala * cos(theta)

    No Pygame, o eixo y cresce para baixo.
    """

    pivo_x, pivo_y = pivo

    massa_x = pivo_x + L * escala * math.sin(theta)
    massa_y = pivo_y + L * escala * math.cos(theta)

    # Linha vertical de referência
    pygame.draw.line(
        tela,
        cor_referencia,
        (pivo_x, pivo_y),
        (pivo_x, int(pivo_y + L * escala)),
        1
    )

    # Haste
    pygame.draw.line(
        tela,
        cor_haste,
        (pivo_x, pivo_y),
        (int(massa_x), int(massa_y)),
        4
    )

    # Pivô
    pygame.draw.circle(
        tela,
        cor_haste,
        (pivo_x, pivo_y),
        7
    )

    # Massa
    pygame.draw.circle(
        tela,
        cor_massa,
        (int(massa_x), int(massa_y)),
        23
    )
