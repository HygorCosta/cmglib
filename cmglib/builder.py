""" Read and write properties from CMG files."""
import re
import numpy as np
from util import parse_all_string
from tqdm import tqdm


class Builder:

    def __init__(self, nx:int, ny:int, nz:int) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def ler_propriedades(self, arq_entrada, dtype=float, shape=None):
        """ Ler arquivo de propriedades no formato CMG."""
        saida_dados = []
        with open(arq_entrada, 'r+', encoding='UTF-8') as entrada:
            linha = entrada.readline()
            inicio_dados = re.compile(r'\s*\d.*')
            while linha != '':
                match = inicio_dados.match(linha)
                if match:
                    entrada.seek(entrada.tell()-len(linha))
                    all_string = entrada.read()
                    saida_dados = parse_all_string(all_string, dtype, shape)
                linha = entrada.readline()
        return saida_dados

    def ler_coord(self, arq_entrada):
        """Lê um arquivo de dados do tipo COORD.

        Args:
            arq_entrada (str): arquivo include com os dados COORD

        Returns:
            numpy.ndarray: vetor de saida com shape ((nx+1)*(ny+1), 6),
            dados do tipo float.
        """
        # num_cel = int((self.nx+1)*(self.ny+1)*6)
        print(f'\nLendo arquivo: {arq_entrada}')
        shape = ((self.ny+1)*(self.nx+1), 6)
        saida_dados = self.ler_propriedades(arq_entrada, shape=shape)
        return saida_dados

    def ler_zcorn(self, arq_entrada):
        """Lê um arquivo de dados do tipo ZCORN,

        Args:
            arq_entrada (str): caminho para o arquivo include ZCORN

        Returns:
            ztop (np.ndarray): vetor de profundidade para cada célula
              com (4*nx*ny*nz) valores - nós de topo
            zbase (np.ndarray): vetor de profundiade para cada célula
              com (4*nx*ny*nz) valores - nós de base
        """
        # num_cel = int(self.nx*self.ny*self.nz*8)
        print(f'\nLendo arquivo {arq_entrada}...')
        saida_dados = self.ler_propriedades(arq_entrada)
        ztop = np.empty([0])
        zbase = np.empty([0])
        for i in range(self.nz):
            # NW-T and NE-T
            inicio_1 = int(2*i*self.nx*self.ny*4)
            # SW-T and SE-T
            fim_1 = int((2*i+1)*self.nx*self.ny*4)
            # NW-B and NE-B
            inicio_2 = fim_1
            # SW-B and SE-B
            fim_2 = int((2*i+2)*self.nx*self.ny*4)
            ztop = np.append(ztop, saida_dados[inicio_1:fim_1])
            zbase = np.append(zbase, saida_dados[inicio_2:fim_2])
        return ztop, zbase

    def retornar_valores_formato_zcorn(self, coord):
        saida = np.zeros((self.nx*self.ny*4))
        z = 0
        range_j = [i for i in range(self.ny+1) for _ in range(2)][1:-1]
        for j in range_j:
            for i in range(self.nx+1):
                if (i==0) or (i==self.nx):
                    saida[z] = coord[i + j*(self.nx+1)]
                    z += 1
                else:
                    saida[z] = coord[i + j*(self.nx+1)]
                    saida[z+1] = coord[i + j*(self.nx+1)]
                    z += 2
        saida_concat = saida
        for i in range(self.nz-1):
            saida_concat = np.append(saida_concat, saida)
        return saida_concat

    def retornar_xyz(self, coord, ztop, zbot):
        pilars_xyz_1 = coord[:,:3]
        pilars_xyz_2 = coord[:, 3:]
        delta_xyz = pilars_xyz_2 - pilars_xyz_1
        pilar_x = self.retornar_valores_formato_zcorn(pilars_xyz_1[:, 0])
        pilar_y = self.retornar_valores_formato_zcorn(pilars_xyz_1[:, 1])
        pilar_z = self.retornar_valores_formato_zcorn(pilars_xyz_1[:, 2])
        delta_x = self.retornar_valores_formato_zcorn(delta_xyz[:, 0])
        delta_y = self.retornar_valores_formato_zcorn(delta_xyz[:, 1])
        delta_z = self.retornar_valores_formato_zcorn(delta_xyz[:, 2])
        delta_ztop = ztop - pilar_z
        delta_zbot = zbot - pilar_z

        coor_x = (delta_x/delta_z) * delta_ztop + pilar_x
        coor_x = np.append(coor_x, (delta_x/delta_z) * delta_zbot + pilar_x)
        coor_y = (delta_y/delta_z) * delta_ztop + pilar_y
        coor_y = np.append(coor_y, (delta_y/delta_z) * delta_zbot + pilar_y)
        coor_z = np.append(ztop, zbot)

        self.coor = {'x':coor_x, 'y':coor_y, 'z':coor_z}

    def retornar_vertices_celula(self, matriz, i, j, k):
        """Retorna os vertices da celula i,j,k.

            3---------2
           /|        /|
          / |       / |
         0---------1  |
         |  7------|--6
         | /       | /
         |/        |/
         4---------5

        Args:
            matriz (np.ndarray): matriz com todas as coordenadas no formato COORD
            para cada direcao
            i (int): indice da celula na direcao x
            j (int): indice da celula na direcao y
            k (int): indice da celula na direcao z

        Returns:
            np.ndarray: matriz com as coordenadas dos oito vertices das celulas
        """
        pos1 = k*self.nx*self.ny*4 + j*self.nx*4 + i*2
        pos2 = k*self.nx*self.ny*4 + j*self.nx*4 + i*2 + 1
        pos3 = k*self.nx*self.ny*4 + self.nx*2 + j*self.nx*4 + i*2 + 1
        pos4 = k*self.nx*self.ny*4 + self.nx*2 + j*self.nx*4 + i*2
        (n,) = matriz.shape
        n = int(n/2)
        return np.array([matriz[pos1], matriz[pos2], matriz[pos3], matriz[pos4],
                         matriz[pos1+n], matriz[pos2+n], matriz[pos3+n], matriz[pos4+n]])

    def calculate_volumes(self):
        """Calculate the volume of each cell.
        """
        ncelulas = self.nx * self.ny * self.nz
        pbar = tqdm(total=ncelulas, desc='Loading Volumes', position=0)
        self.volumes = np.zeros((self.nx, self.ny, self.nz))
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    x = self.retornar_vertices_celula(self.coor['x'], i, j, k)
                    y = self.retornar_vertices_celula(self.coor['y'], i, j, k)
                    z = self.retornar_vertices_celula(self.coor['z'], i, j, k)

                    u = np.array((x[1], y[1], z[1])) - np.array((x[0], y[0], z[0]))
                    v = np.array((x[3], y[3], z[3])) - np.array((x[0], y[0], z[0]))
                    w = np.array((x[5], y[5], z[5])) - np.array((x[0], y[0], z[0]))

                    V = np.linalg.det(np.array((u, v, w)))
                    self.volumes[i, j, k] = np.abs(V)
                pbar.update(self.nx)
