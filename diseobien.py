# ================================ BLOQUE DE FUNCION, NO MODIFICAR =============================

import math as ma
from tabulate import tabulate
# pip install tabulate

def escribir_en_txt(texto):
    try:
        with open("informe.txt", 'a') as archivo:
            archivo.write("\n\n\n")
            archivo.write(texto)
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")

def borrar():
    try:
        with open("informe.txt", 'w') as archivo:
            archivo.write("=====================     INFORME DE CALCULO DE TRANSFORMADOR     =====================")
    except Exception as e:
        print(f"Error al borrar el archivo: {e}")

def alambre(awg):
    tabla = [
        #AWGF, Seccion mm2, diametro int mm, diametro ext mm, aislamiento mm
        [24, 0.205, 0.511, 0.580, 0.069],
        [23, 0.250, 0.573, 0.643, 0.070],
        [22, 0.325, 0.644, 0.714, 0.070],
        [21, 0.411, 0.723, 0.798, 0.075],
        [20, 0.518, 0.812, 0.892, 0.080],
        [19, 0.653, 0.912, 0.993, 0.081],
        [18, 0.823, 1.024, 1.110, 0.086],
        [17, 1.039, 1.150, 1.240, 0.090],
        [16, 1.309, 1.291, 1.380, 0.089],
        [15, 1.651, 1.450, 1.550, 0.100],
        [14, 2.001, 1.628, 1.730, 0.102],
        [13, 2.624, 1.828, 1.923, 0.102],
        [12, 3.310, 2.053, 2.150, 0.097],
        [11, 4.172, 2.305, 2.408, 0.103],
        [10, 5.260, 2.588, 2.695, 0.107],
        [9,  6.633, 2.906, 3.020, 0.114],
        [8,  8.376, 3.264, 3.380, 0.140],
        [7, 10.550, 3.665, 3.787, 0.122],
    ]
    for fila in tabla:
        if fila[0] == awg:
            return [fila[0], fila[1]/1e6, fila[2]/1e3, fila[3]/1e3, fila[4]/1e3]
    raise ValueError(f"AWG {awg} no está en la tabla.")

def distancias(V_dist):
    V_dist /= 1000
    trans_dist = [
        [6, 6, 12, 4, 2.0, 4.0, 10.0, 12],
        [11, 10, 15, 4, 2.5, 4.0, 10.5, 15],
        [22, 10, 25, 4, 3.0, 4.0, 11.0, 15],
        [33, 10, 40, 5, 5.0, 5.0, 15.0, 20],
        [50, 10, 50, 5, 5.0, 8.0, 18.0, 25],
    ]
    for i in range(len(trans_dist)):
        if trans_dist[i][0] > V_dist:
            return trans_dist[i - 1]
    raise ValueError(f"No se encontró distancia para KV_dist={V_dist*1000}V")

def pletina(tipo):
    try:
        espesor, ancho = tipo
    except ValueError:
        raise ValueError("El tipo de pletina debe ser una lista o tupla de dos elementos [espesor, ancho].")

    tabla = [   [   0,  1.4,  1.6,  1.8,  2,    2.2,  2.5,  2.8,  3,    3.5,  4,    4.5,  5,    5.5,  6  ], # ... espesor
                [ 4.5,  5.8,  6.5,  7.4,  8.3,  9.2, 10.6, 11.9, 12.8, 14.9, 17.1, 19.4, 21.6, 23.7, 26.0],
                [ 5.0,  6.5,  7.3,  8.3,  9.3, 10.3, 11.8, 13.3, 14.3, 16.6, 19.1, 21.6, 24.1, 26.5, 29.0],
                [ 5.5,  7.2,  8.1,  9.2, 10.3, 11.4, 13.1, 14.7, 15.8, 18.4, 21.1, 23.9, 26.6, 29.2, 32.0],
                [ 6.0,  7.9,  8.9, 10.1, 11.3, 12.5, 14.3, 16.1, 17.3, 20.1, 23.1, 26.1, 29.1, 32.0, 35.0],
                [ 6.5,  8.6,  9.7, 11.0, 12.3, 13.6, 15.6, 17.5, 18.8, 21.5, 25.1, 28.4, 31.6, 34.7, 38.0],
                [ 7.0,  9.3, 10.5, 11.9, 13.3, 14.7, 16.8, 18.9, 20.3, 23.6, 27.1, 30.6, 34.1, 37.5, 41.0],
                [ 7.5, 10.0, 11.3, 12.8, 14.3, 15.8, 18.1, 20.3, 21.8, 25.4, 29.1, 32.9, 36.6, 40.2, 44.0],
                [ 8.0, 10.7, 12.1, 13.7, 15.3, 16.9, 19.3, 21.7, 23.3, 27.1, 31.1, 35.1, 39.1, 43.0, 47.0],
                [ 8.5, 11.4, 12.9, 14.6, 16.3, 18.0, 20.6, 23.1, 24.8, 28.9, 33.1, 37.4, 41.6, 45.7, 50.0],
                [ 9.0, 12.1, 13.7, 15.5, 17.3, 19.1, 21.8, 24.5, 26.3, 30.6, 35.1, 39.6, 44.1, 48.5, 53.0],
                [ 9.5, 12.8, 14.5, 16.4, 18.3, 20.2, 23.1, 26.0, 27.9, 32.4, 37.1, 41.6, 46.1, 51.2, 56.0],
                [10.0, 13.5, 15.3, 17.3, 19.3, 21.3, 24.3, 27.3, 29.3, 34.1, 39.1, 44.1, 49.1, 54.0, 59.0],
                [10.5, 14.2, 16.1, 18.2, 20.3, 22.4, 25.6, 28.7, 30.7, 35.9, 41.1, 46.4, 51.6, 56.7, 62.0],
                [11.0, 14.9, 16.9, 19.1, 21.3, 23.5, 26.8, 30.1, 32.3, 37.6, 43.1, 48.6, 54.1, 59.5, 65.0],
                [11.5, 15.6, 17.7, 20.0, 22.3, 24.6, 28.1, 31.5, 33.8, 39.4, 45.1, 50.9, 56.6, 62.2, 68.0],
                [12.0, 16.3, 18.5, 20.9, 23.3, 25.7, 29.2, 32.9, 35.3, 41.1, 47.1, 53.1, 59.1, 65.0, 71.0],
                [12.5, 17.0, 19.3, 21.8, 24.3, 26.8, 30.6, 34.3, 36.8, 43.1, 49.1, 55.4, 61.6, 67.7, 74.0],
                [13.0, 17.7, 20.1, 22.7, 25.3, 27.9, 31.8, 35.7, 38.3, 44.6, 51.1, 57.6, 64.1, 70.5, 77.0],
                [13.5, 18.4, 20.9, 23.6, 26.3, 29.0, 33.1, 37.1, 39.8, 46.4, 53.1, 59.9, 66.6, 73.2, 80.0],
                [14.0, 19.1, 21.7, 24.5, 27.3, 30.1, 34.3, 38.5, 41.3, 48.1, 55.1, 61.9, 68.7, 76.0, 83.0],
                [14.5, 19.8, 22.5, 25.4, 28.3, 31.2, 35.6, 39.9, 42.8, 49.9, 57.1, 64.4, 71.6, 78.7, 86.0],
                [15.0, 20.5, 23.3, 26.4, 29.4, 32.3, 36.8, 41.4, 44.3, 51.6, 59.1, 66.6, 74.1, 81.5, 89.0],
                [15.5, 21.2, 24.1, 27.2, 30.3, 33.4, 38.1, 42.7, 45.8, 53.4, 61.1, 68.9, 76.6, 84.2, 92.0],
                [16.0, 21.9, 24.9, 28.1, 31.3, 34.5, 39.4, 44.1, 47.3, 55.1, 63.1, 71.1, 79.1, 87.0, 95.0],
                [16.5, 22.6, 25.7, 29.0, 32.3, 35.6, 40.6, 45.6, 48.8, 56.9, 65.1, 73.4, 81.6, 89.7, 98.0],
            ]

    try:
        conti = tabla[0].index(espesor) + 1
        contj = next(i for i, row in enumerate(tabla[1:], start=1) if row[0] == ancho)
        area = tabla[contj][conti-1]
    except Exception:
        raise ValueError(f"Pletina con espesor={espesor} y ancho={ancho} no está en la tabla.")

    espesor_aisl = espesor + 0.5
    ancho_aisl = ancho + 0.5
    return [espesor/1000, ancho/1000, espesor_aisl/1000, ancho_aisl/1000, area/1000000]

def escalones(esc1):
    escal = [0.6366, 0.7869, 0.8582, 0.8862, 0.9085, 0.9217]
    if 1 <= esc1 <= len(escal):
        return escal[esc1-1]
    else:
        raise ValueError("Índice de escalones fuera de rango (1-6)")

def tensiones(Snom, Vnom, Grup):
    """Calcula tensiones e intensidades por devanado."""
    try:
        # Se acepta especificar el grupo completo, por ejemplo "Dyn5" o "Yd11".
        if Grup.startswith("Dy"):
            Vbt = Vnom[1] / ma.sqrt(3)
            Vat0 = Vnom[0] * (1 - tap)
            Vat1 = Vnom[0]
            Vat2 = Vnom[0] * (1 + tap)
        elif Grup.startswith("Yd"):
            Vbt = Vnom[1]
            Vat0 = Vnom[0] * (1 - tap) / ma.sqrt(3)
            Vat1 = Vnom[0] / ma.sqrt(3)
            Vat2 = Vnom[0] * (1 + tap) / ma.sqrt(3)
        else:
            raise ValueError(f"Grupo de conexión no válido: {Grup}")

        Ibt = Snom / Vbt
        Iat = [Snom / Vat0, Snom / Vat1, Snom / Vat2]
        Vat = [Vat0, Vat1, Vat2]
        return Vbt, Vat, Ibt, Iat

    except ZeroDivisionError:
        raise ZeroDivisionError("División por cero al calcular tensiones. Revisa los valores de entrada.")

def analisis_cond(plet_alam, tipo_cond, junt_sup, Ibt, Iat):
    try:
        area = []
        densi_I = []
        ancho = []
        espesor = []
        for cont, tipo in enumerate(plet_alam):
            if tipo == 0:
                ala = alambre(tipo_cond[cont])
                area.append(junt_sup[cont][0] * junt_sup[cont][1] * ala[1])
                ancho.append(junt_sup[cont][0] * ala[3])
                espesor.append(junt_sup[cont][1] * ala[3])
            elif tipo == 1:
                plet = pletina(tipo_cond[cont])
                area.append(junt_sup[cont][0] * junt_sup[cont][1] * plet[4])
                ancho.append(junt_sup[cont][0] * plet[3])
                espesor.append(junt_sup[cont][1] * plet[2])
            else:
                raise ValueError(f"Tipo de conductor inválido: {tipo}")
        densi_I.append(Iat[0] / area[0] / 1e6)
        densi_I.append(Ibt / area[1] / 1e6)
        return area, densi_I, ancho, espesor
    except Exception as e:
        raise RuntimeError(f"Error en analisis_cond: {e}")

def nucleo(Sfe, esc, fa):
    try:
        S_nuc_tot = Sfe / fa
        S_nuc_circ = S_nuc_tot / escalones(esc)
        D_nuc_circ = ma.sqrt(S_nuc_circ * 4 / ma.pi)
        D_nuc_circ_aisl = D_nuc_circ + 6/1000
        return S_nuc_circ, D_nuc_circ, D_nuc_circ_aisl
    except Exception as e:
        raise RuntimeError(f"Error en nucleo: {e}")

def Espiras_capas(Sfe, B, f, Vbt, Vat, N_cap):
    try:
        VxT = 4.44 * B * Sfe * f
        Nat = round(Vat[2]/VxT  , 0)
        Nat0 = round(Vat[0]/VxT  , 0)
        Nbt = round(Vbt/VxT , 0)
        N = [Nat, Nbt]
        DifN = Nat - Nat0
        Res_N = [Nat - Vat[2]/VxT , Nbt - Vbt/VxT]
        N_espxcap = [round((Nat + 2) / N_cap[0] + 0.4999999), round((Nbt + 2) / N_cap[1] + 0.4999999)]
        Res_espxcap = [N_espxcap[0] - (Nat + 2) / N_cap[0], N_espxcap[1] - (Nbt + 2) / N_cap[1]]
        return N, DifN, Res_N, N_espxcap, Res_espxcap
    except Exception as e:
        raise RuntimeError(f"Error en Espiras_capas: {e}")

def aisl_entrecapas(tipo_cond, Vat0, Vbt, N_cap):
    try:
        N_cap_at = N_cap[0]
        N_cap_bt = N_cap[1]
        Vat = Vat0[2]*2
        const = 1 / 3e6
        V_cap = [Vat / (N_cap_at - 1), Vbt*2 / (N_cap_bt - 1) if N_cap_bt > 1 else 0]
        aisl_ext = alambre(tipo_cond[0])[4]
        aisl_cap0 = [V_cap[0] * const - aisl_ext, V_cap[1] * const - 0.5 / 1000]
        aisl_cap = [max(0, x) for x in aisl_cap0]
        return aisl_cap
    except Exception as e:
        raise RuntimeError(f"Error en aisl_entrecapas: {e}")

def Longitud_axial(N_espxcap, ancho_ext, Vat, Vbt):
    try:
        long_ax = [N_espxcap[i] * ancho_ext[i] for i in range(2)]
        Alt_col = [long_ax[0] + 2*distancias(Vat[2])[1]/1000, long_ax[1] + 2*distancias(Vbt)[1]/1000]
        return long_ax, Alt_col
    except Exception as e:
        raise RuntimeError(f"Error en Longitud_axial: {e}")

def Calc_diam_peso_CU(D_nuc_circ_aisl, espesor_ext, N_cap, aisl_cap, area, N, dens_cu, Canal):
    try:
        Dint_bt = D_nuc_circ_aisl
        Dext_bt = Dint_bt + espesor_ext[1] * N_cap[1] * 2 + 2 * (N_cap[1] - 1) * aisl_cap[1]
        Dint_at = Dext_bt + Canal*2
        Dext_at = Dint_at + espesor_ext[0] * N_cap[0] * 2 + 2 * (N_cap[0] - 1) * aisl_cap[0]
        Dmed = [(Dint_at + Dext_at) / 2, (Dint_bt + Dext_bt) / 2]
        diame = [Dint_bt, Dext_bt, Dint_at, Dext_at]
        sobred=[1.05,  1.12] # per derivaciones y otros
        pesoCU = [N[i] * Dmed[i] * ma.pi * area[i] * sobred[i] * dens_cu * 3 for i in range(2)]
        return diame, pesoCU
    except Exception as e:
        raise RuntimeError(f"Error en Calc_diam_peso_CU: {e}")

def peso_fe(Alt_col, Vat, diame, Sfe, dens_fe):
    try:
        d = distancias(Vat[2])[7]/1000
        Alt_col_max = max(Alt_col)
        Lm_nuc = 3 * Alt_col_max + 4 * diame[3] + 4 * d + 2 * diame[0]
        peso_nuc = Lm_nuc * Sfe * dens_fe
        return peso_nuc
    except Exception as e:
        raise RuntimeError(f"Error en peso_fe: {e}")


def Perd_fierro(f, B, peso_nuc):
    valores_50Hz = [0.00183, 0.00702, 0.0152, 0.0265, 0.0400, 0.0564, 0.0753, 0.0968, 0.121, 0.148, 0.179, 0.214, 0.253, 0.298, 0.353, 0.418, 0.514, 0.658, 0.770]
    valores_60Hz = [0.00242, 0.00928, 0.0202, 0.0347, 0.0528, 0.0742, 0.0990, 0.127, 0.159, 0.195, 0.236, 0.281, 0.333, 0.391, 0.462, 0.546, 0.666, 0.845, 0.990]
    valores = [valores_50Hz, valores_60Hz]
    i = 1 if f == 60 else 0

    for x in range(18):
        flujo_T = (1 + x) / 10
        if flujo_T >= B:
            break
    else:
        x = 17
    try:
        Pe_fe = valores[i][x] + (valores[i][x+1] - valores[i][x]) * (B - flujo_T) / 0.1
    except IndexError:
        Pe_fe = valores[i][-1]

    Perd_nuc = Pe_fe * peso_nuc
    return Perd_nuc

def Perd_cobre(PesoCu, densi_I):
    Perd_cu = []
    try:
        for x in range(2):
            Perd_cu.append((2.427 * densi_I[x]**2) * PesoCu[x])
    except Exception as e:
        raise RuntimeError(f"Error al calcular pérdidas de cobre: {e}")
    return Perd_cu

def formatear_celda(x):
    if isinstance(x, float):  # Número con decimales
        if abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0):  # Notación científica
            return f"{x:.2e}"
        else:
            return f"{x:.2f}"
    elif isinstance(x, list):  # Lista -> convertir a string
        y2=[]
        for y in x:
            y2.append(formatear_celda(y))
        return str(y2)
    else:  # Texto, enteros, etc.
        return str(x)

def tcc(Vat, Iat, N, DifN, diame, canal, long_ax, freq):
    Rat=(diame[3]-diame[2])/2
    Rbt=(diame[1]-diame[0])/2
    dmed=(diame[3]+diame[0])/2
    b=Rat+Rbt+canal
    gap=Rat/3+Rbt/3+canal
    L=long_ax[0]
    k=1-(b/(ma.pi*L))*(1-ma.e**(-ma.pi*L/b))
    Ls=L/k
    Us=2.48*10**-5*(N[0]-DifN/2)**2*Iat[1]*gap*dmed*freq/Ls
    vcc=Us/Vat[1]*100
    return round(vcc,3)

def calentamiento(Perd_cu, Perd_nuc, N_cap, aisl_cap, diame, tipo_cond, long_ax, D_nuc_circ):
    """Calcula parámetros térmicos y dimensiones del tanque.

    Devuelve la longitud de aletas necesaria, las dimensiones del
    tanque (largo y ancho), el área de radiación, el área desarrollada
    de convección, el número de aletas y una lista de temperaturas.
    """
    temp=30 # temperatura ambiente.
    Ptot=Perd_nuc+sum(Perd_cu)
    Dmed=(diame[3]+diame[2])/2
    Amed=3.1416*Dmed*long_ax[0] # area media de disipacion de calor de bobina de AT (BT no importa)
    c=0.2 # Coeficiente e discipacion de calor
    al=alambre(tipo_cond[0]) # datos del alambre
    dint=al[2] # diametro interno del alambre en metros
    dext=al[3] # diametro externo del alambre en metros 
    Tmax=65+temp # tmepratura maxiam del cobre
    e=(aisl_cap[0]+dext-1/4*(dint**2/(aisl_cap[0]+dext))) # calculo de la equivalente del espesor entre capas, en metros
    T2=Tmax-(N_cap[0]/3+1/(6*N_cap[0]))*(e*0.8*Perd_cu[0]/(3*N_cap[0]))/(c*Amed) # temperatura en el exterior del devanado de AT
# diferencia de temperatura en el aceite
    # alfa*calent = P/A
    altura_PA=long_ax[0]+2*D_nuc_circ
    A_pa=4*diame[3]*3.1416*(altura_PA) # Area externa apriximada de parte activa
    cal=0
    for i in range(100000000):
        cal=i*2000/100000000
        P_A=Ptot/A_pa
        coef_c=-0.0276*cal**2 + 2.6458*cal + 63.644
        coef_cal=coef_c*cal
        if  abs(P_A-coef_cal)<1: # se halla el valor de "cal" que permite cumplir con esta igualdad de P_A=Coef_cal (0)
            break
    if (i==100000000-1):
        print("Debes corregir tu diseño.")
    T3=T2-cal
# diferencia de temperatura con el exterior
    Largo_t=3*diame[3]+100/1000 # Largo del transformador 
    Ancho_t=diame[3]+100/1000 # ancho del transformador
    Alt_t=altura_PA # altura de la parte activa = altura referencial de las aletas
    N_aletas=2*(round(Largo_t/0.050-0.4999,0)+1)+2*(round(Ancho_t/0.050-0.4999,0)+1) # cantidad de aletas
    cal2=T3-temp
    Coef1=0.0288*cal2+5.281
    Coef2=-0.0028*cal2**2+0.1613*cal2+3.0412
    Coef_calt=(Coef1+Coef2)*cal2
    for i in range(100000):
        L_aletas = 0.25 * i / 100000  # hasta 25 cm como máximo
        A_ext = Alt_t * (2 * Largo_t + 2 * Ancho_t + 4 * ma.sqrt(L_aletas**2 + L_aletas**2))
        A_des = Alt_t * (2 * Largo_t + 2 * Ancho_t + N_aletas * L_aletas * 2)
        P_A2 = Ptot / (A_ext + A_des)
        if abs(P_A2 - Coef_calt) < 1:
            break
    if i == 100000 - 1:
        print("La longitud de las letas sale por encima de 25 centimetros, debes de corregir tu diseño")
    return L_aletas, Largo_t, Ancho_t, A_ext, A_des, N_aletas, [Tmax, T2, T3, temp]

# ================================ BLOQUE PRINCIPAL =============================
# ESTO ES LO QUE USTED TIENE QUE CAMBIAR, ESTA BIEN QUE PREGUNTE, PERO NO EXAGERE

# if __name__ == "__main__":
#     try:
        # DATOS DE ENTRADA
Snom = 100 * 1000 / 3  # potencia monofásica en VA
Vnom = [10000, 230]  # tensiones en voltios
Grupo = "Dyn5"  # grupo de conexión
tap = 0.05  # +/- 5 % de TAP
f = 60

# CONSTANTES
fa = 0.95 # factor de apilamiento
dens_cu = 8.9 * 1000
dens_fe = 7.85 * 1000
B = 1.6
Canal = 13 / 1000 # de dispersión

# VARIABLES
Sfe = 330 / 10000 # sección útil de nucleo 
esc = 5 # cantidad de escalones
N_cap = [7, 1] # cantidad de capas [AT, BT]
plet_alam = [0, 1] # 0 para alambre, 1 para pletina # [AT, BT]
# tipo_cond = [9, [3, 12.5]] # para la pletina= [espesor(menor), ancho(mayor)]
tipo_cond = [9, [3, 13.5]] # para la pletina= [espesor(menor), ancho(mayor)]
junt_sup = [[1, 1], [2, 6]]


# CÁLCULOS
Vbt, Vat, Ibt, Iat = tensiones(Snom, Vnom, Grupo)
area, densi_I, ancho_ext, espesor_ext = analisis_cond(plet_alam, tipo_cond, junt_sup, Ibt, Iat)
S_nuc_circ, D_nuc_circ, D_nuc_circ_aisl = nucleo(Sfe, esc, fa)
N, DifN, Res_N, N_espxcap, Res_espxcap = Espiras_capas(Sfe, B, f, Vbt, Vat, N_cap)
aisl_cap = aisl_entrecapas(tipo_cond, Vat, Vbt, N_cap)# despues del vidceo hice un cambio aqui, era al doble de tnesion
long_ax, Alt_col = Longitud_axial(N_espxcap, ancho_ext, Vat, Vbt)
diame, pesoCU = Calc_diam_peso_CU(D_nuc_circ_aisl, espesor_ext, N_cap, aisl_cap, area, N, dens_cu, Canal)
peso_nuc = peso_fe(Alt_col, Vat, diame, Sfe, dens_fe)
Perd_nuc = Perd_fierro(f, B, peso_nuc)
Perd_cu = Perd_cobre(pesoCU, densi_I)
vcc = tcc(Vat, Iat, N, DifN, diame, Canal, long_ax, f)
L_aletas, Largo_t, Ancho_t, A_ext, A_des, N_aletas, temperaturas = calentamiento(
    Perd_cu, Perd_nuc, N_cap, aisl_cap, diame, tipo_cond, long_ax, D_nuc_circ
)

borrar()
# SALIDA DE DATOS - NUCLEO
sal_nuc=[]
sal_nuc.append([Sfe*10000, "Valor ingresado de seccion de núcleo en cm2"])
sal_nuc.append([S_nuc_circ*10000, "sección de circunferencia cincundate al nucleo" ])
sal_nuc.append([D_nuc_circ*100, "Diametro de la circunferencia circundante al nucleo en cm"])
sal_nuc.append([Alt_col, "Altura de columna del núcleo en m, AMBOS VALORES DEBEN SER IGUALES"])
sal_nuc.append([peso_nuc, "Peso del nucleo en Kg"])
sal_nuc.append([Perd_nuc, "Pérdias en el nucleo en Watts a tensión nominal"])
encabezados_nuc = ["NUCLEO FERROMAGNETICO", "Descripcion"]
tabla_fmt_nuc = [[formatear_celda(celda) for celda in fila] for fila in sal_nuc]
impri_nuc=tabulate(tabla_fmt_nuc, headers=encabezados_nuc, tablefmt="grid")
#print(impri_nuc)
escribir_en_txt(impri_nuc)


# SALIDA DE DATOS - BOBINA
sal_bob=[]
sal_bob.append([N, "Espiras eléctricas por devanado"])
sal_bob.append([[N[0]+2, N[1]+2], "Espiras mecánicas por devanado"])
sal_bob.append([Res_N, "Residuo de cantidad de espiras totales, deberia ser mucho menor a 0.5"])
sal_bob.append([DifN, "Diferencia de primer y ultimo tap en devanado de AT MENOR QUE ESPIRAS POR CAPA"])
sal_bob.append([N_cap, "Numero de capas definido por el usuario"])
sal_bob.append([Res_espxcap, "Residuo de cantidad de espiras por capa, deberia ser meno al 5% de la cantidad de espirtas por capa solo en AT"])
sal_bob.append([plet_alam, "tipo de condcutor, 0 para alambre y 1 para pletina"])
sal_bob.append([tipo_cond, "numero AWG para alambres y [espesor,ancho] para pletina"])
sal_bob.append([junt_sup, "Configurtaciond e juntas y superpuestas"])
sal_bob.append([[espesor_ext[0]*1000, espesor_ext[1]*1000], "Grosor de los conductores totales en mm"])
sal_bob.append([[ancho_ext[0]*1000, ancho_ext[1]*1000], "Largo de los conductores totales en mm"])
sal_bob.append([[area[0]*1000000, area[1]*1000000], "Sección transversal de los conductores totales en mm2"])
sal_bob.append([N_espxcap, "Numero de espiras mecánicas por capa"])
sal_bob.append([long_ax, "Altura de cada bobina en metros"])
sal_bob.append([densi_I, "Densidad de corriente A/mm2 LAS 2 ENTRE 3 Y 3.5"])
sal_bob.append([[aisl_cap[0]*1000, aisl_cap[1]*1000], "Aislamiento entre capas en mm"])
sal_bob.append([Canal*1000, "Canal de dispersión en mm"])
sal_bob.append([[diame[2]*100, diame[0]*100], "Diametros internos en centimetros"])
sal_bob.append([[diame[3]*100, diame[1]*100], "Diametros externos en centimetros"])
sal_bob.append([[pesoCU[0], pesoCU[1]], "Pesos por bobina  bobinas en Kg"])
sal_bob.append([Perd_cu, "Perdidas en el cobre (par alas 3 bobinas) a plena carga"])

encabezados_bob = ["BOBINA AT / BOBINA BT", "Descripcion"]
tabla_fmt_bob = [[formatear_celda(celda) for celda in fila] for fila in sal_bob]
impri_bob=tabulate(tabla_fmt_bob, headers=encabezados_bob, tablefmt="grid")
#print(impri_bob)
escribir_en_txt(impri_bob)

# SALIDA DE DATOS - TENSIÓN DE CORTOCIRCUITO
sal_tcc=[]
sal_tcc.append([vcc, "%"])


encabezados_tcc = ["TENSIÓN de CC", "Descripcion"]
tabla_fmt_tcc = [[formatear_celda(celda) for celda in fila] for fila in sal_tcc]
impri_tcc=tabulate(tabla_fmt_tcc, headers=encabezados_tcc, tablefmt="grid")
#print(impri_bob)
escribir_en_txt(impri_tcc)
    # except Exception as e:
    #     print(f"Error durante la ejecución del programa: {e}")

# SALIDA DE DATOS - TEMPERATURA
sal_temp=[]
sal_temp.append([L_aletas, "Longitu de aletas (MENOR A 25cm  o 0.25m)"])
sal_temp.append([N_aletas, "Numero de aletas"])
sal_temp.append([Largo_t, "Longitud del transformador en metros (sin contar aletas)"])
sal_temp.append([Ancho_t, "Ancho del transformador en metros (sin contar aletas)"])
sal_temp.append([A_ext, "Area externa de radiacion en m2"])
sal_temp.append([A_des, "Area desarrollada de conveccion en m2"])
sal_temp.append([temperaturas, "temperatruas °C en el transformador [bob AT, en la superficie de PA, en el tanque, en el exterior(Tamb) ]"])


encabezados_temp = ["DATOS DEL TANQUE (refrigeracion)", "Descripcion"]
tabla_fmt_temp = [[formatear_celda(celda) for celda in fila] for fila in sal_temp]
impri_temp = tabulate(tabla_fmt_temp, headers=encabezados_temp, tablefmt="grid")
#print(impri_bob)
escribir_en_txt(impri_temp)
