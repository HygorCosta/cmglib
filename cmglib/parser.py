""" Read and write properties from CMG files."""
import re
import numpy as np
import xarray as xr
from tqdm import tqdm


class CMG:

    def __init__(self, nx:int, ny:int, nz:int) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz

    @staticmethod
    def ler_dados(entrada, ncelulas=0):
        """ Ler propriedade no formato CMG iniciando diretamente nos valores numéricos.

        Args:
            entrada (fileObject): arquivo na aberto.

        Returns:
            saida_dados: lista com conjunto de dados lidos do arquido 'entrada'
        """
        pbar = tqdm(total=ncelulas, desc='Loading', position=0)
        delta_cont = 0
        cont = 0
        update = ncelulas/100
        saida_dados = []
        linha = entrada.readline()
        while linha != '':
            linha = linha.split()
            for dado in linha:
                if '*' not in dado:
                    saida_dados.append(dado)
                    delta_cont += 1
                else:
                    dado = dado.split('*')
                    mult, value = dado[0], dado[1]
                    saida_dados.extend([value]*int(mult))
                    delta_cont += mult
            cont += delta_cont
            if (ncelulas > 0 and delta_cont >= update) or cont==ncelulas:
                pbar.update(delta_cont)
                delta_cont = 0
            linha = entrada.readline()
        if ncelulas > 0:
            pbar.close()
        return saida_dados

    def ler_propriedades(self, arq_entrada, ncelulas=0):
        """ Ler arquivo de propriedades no formato CMG."""
        saida_dados = []
        with open(arq_entrada, 'r+', encoding='UTF-8') as entrada:
            linha = entrada.readline()
            inicio_dados = re.compile(r'\s*\d.*')
            while linha != '':
                match = inicio_dados.match(linha)
                if match:
                    entrada.seek(entrada.tell()-len(linha))
                    saida_dados = self.ler_dados(entrada, ncelulas)
                    break
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
        num_cel = int((self.nx+1)*(self.ny+1)*6)
        print(f'\nLendo arquivo: {arq_entrada}')
        saida_dados = self.ler_propriedades(arq_entrada, num_cel)
        return np.array(saida_dados).reshape((self.ny+1)*(self.nx+1), 6).astype(float)

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
        num_cel = int(self.nx*self.ny*self.nz*8)
        print(f'\nLendo arquivo {arq_entrada}...')
        saida_dados = self.ler_propriedades(arq_entrada, num_cel)
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
            ztop = np.append(ztop, np.array(saida_dados[inicio_1:fim_1]).astype(float))
            zbase = np.append(zbase, np.array(saida_dados[inicio_2:fim_2]).astype(float))
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
