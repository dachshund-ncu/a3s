#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# -- importujemy potrzebne moduły --
# -- numpy --
from numpy import exp, sin, cos, asarray, sqrt, mean, pi, radians, zeros, inf, nan, hanning, complex128
from numpy.fft import fft
# -----------
# -- math i mpmath --
from math import copysign, floor, ceil
from mpmath import nint
# -----------
# -- astropy --
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
# -------------
# -- sys --
from sys import argv, exit
# ---------
# -- barycorrpy --
from PyAstronomy.pyasl import helcorr
# ----------------

# --------- ZESTAWY GLOBALNYCH FUNKCJI --------------
# ----------------------------------
# -- oblicza prędkość słońca względem LSR (local standard of rest) --
# -- rzutowaną na kierunek, w którym jest źródełko --
# -- RA i DEC podawane muszą być w STOPNIACH --
def lsr_motion(ra,dec,decimalyear):
    # -- zacztnamy --
    vSun0 = 20.0

    # -- współrzędne apeksu słońca z 1900 --
    ras = 18.0 * pi / 12.0 # radiany
    decs = 30.0 * pi / 180.0 # radiany
    # -- obiekt skycoord - apeks słońca w roku 1900 --
    sunc = SkyCoord(ras*u.rad, decs*u.rad, frame=FK5, equinox="B1900")
    # deklarujemy nowy frame z epoką jak nasze obserwacje
    sunc_now = FK5(equinox="J" + str(decimalyear))
    # przechodzimy między frame'ami
    sunc_new = sunc.transform_to(sunc_now)
    # ekstra*ujemy współrzędne i zamieniamy na radiany
    dec_new = radians((sunc_new.dec*u.degree).value)
    ra_new = radians((sunc_new.ra*u.degree).value)
    
    # -- zamieniamy ra i dec na radiany --
    ra = radians(ra)
    dec = radians(dec)
    # -- funkcje na przekazanych współrzędnych
    cdec = cos(dec)
    sdec = sin(dec)

    # -- obliczamy prędkość słońca względem local standard of rest --
    # -- w kierunku ra i dec --
    vSun = vSun0 * ( sin(dec_new) * sdec + cos(dec_new) * cdec * cos(ra-ra_new))
    return vSun
    # -------------


# -- metoda: correctACF --
# nakłada na funkcję autokorelacji krektę na 2 i 3 - poziomową kwantyzację
# przyjmuje w argumencie pojedynczy punkt
def correctACF(autof, r0, rMean):
    # oblicza korekcję do funkcji autokorelacji dla kilku przypadków
    # 3- i 2- poziomowego autokorelatora
    # autof - funkcja autokorelacji (jeden punkt dokładnie)
    # r0 - współczynnik korelacji dla zerowego zapóźnienia
    # bias0 - średni współczynnik dla ogona funkcji autokorelacji (większe zapóźnienia)
    if rMean <= 1e-5:
        # -- limituemy funkcję autokorelacji między 0 i 1
        # tak powinna być znormalizowana (1 w zerowym zapóźnieniu)
        r = min([1.0, abs(autof)])
        if r0 > 0 and r0 < 0.3:
            r = r * 0.0574331 # tak jest w a2s
            rho= r*(60.50861 + r*(-1711.23607 + r*(26305.13517 - r*167213.89458)))/0.99462695104383
            correct_auto = copysign(rho, autof)
            return correct_auto
        elif r0 > 0.3 and r0 < 0.9:
            # trzypoziomoy autokorelator
            r = r * 0.548506
            rho = (r*(2.214 + r*(0.85701 + r*(-7.67838 + r*(22.42186 - r*24.896)))))/0.998609598617374
            correct_auto = copysign(rho, autof)
            return correct_auto
        elif r0 > 0.9:
            rho = sin(1.570796326794897*r)
            correct_auto = copysign(rho, autof)
            return correct_auto
    else:
        autof2 = autof **2.0
        if (abs(autof)) < 0.5:
            fac = 4.167810515925 - r0*7.8518131881775
            a = -0.0007292201019684441 - 0.0005671518541787936*fac
            b =  1.2358980680949918 + 0.03931789097196692*fac
            c = -0.11565632506887912 + 0.08747950965746415*fac
            d =  0.01573239969731158 - 0.06572872697836053*fac
            correct_auto = a + (b + (c + d*autof2)*autof2)*autof
            return correct_auto
        elif (autof > 0.5):
            correct_auto = -1.1568973833585783 + 10.27012449073475*autof - 27.537554958512125*autof2 + 40.54762923890069*autof**3 - 28.758995213769058*autof2**2 + 7.635693826008257*autof**5 + 0.218044850 * (0.53080867 - r0) * cos(3.12416*(autof-0.49721))
            return correct_auto
        else:
            correct_auto = -0.0007466171982634772 + autof*(1.2660000881004778 + autof2*(-0.4237089538779861 + autof2*(1.0910718879007775 + autof2*(-1.452946181286572 + autof2*0.520334578982730)))) + 0.22613222 * (0.53080867 - r0) * cos(3.11635*(autof-0.49595))
            return correct_auto
    

# -- metoda: Clip level --
# -- skopiowana z oryginalnego A2S --
def ClipLevel(ratio):
    Err = 1 - ratio
    x = 0
    for i in range(100):
        dE = Erf(x + 0.01) - Erf(x)
        if dE == 0.0:
            return inf
        x = x + (Err-Erf(x)) * 0.01 / dE
        if abs(Err-Erf(x)) < 0.0001:
            return sqrt(2.0) * x
        else:
            continue

# -- pomocnicza metoda do Clip Level --
def Erf(x):
    t = 1.0 / (1+0.3275911*x)
    erf= 1.0 - t*(0.254829592 + t*(-0.284496736 + t*(1.421413741 + t*(-1.453152027 + t*1.061405429))))*exp(-x**2)
    return erf
# ------------------------------------------------------

# -- deklarujemy klasę pliku .DAT --
class datfile:

    # -- metoda inicjująca klasę --
    def __init__(self, filename):
        # -- stałe --
        self.c = 2.99792458e+5
        self.NN = 8192
        # -- zapisujemy pierwszy atrybut - nazwę pliku .DAT
        self.fname = filename
    
        # -- czytamy dalej --
        self.read_header_and_data()

    # -- metoda wczytująca plik .DAT --
    def read_header_and_data(self):
        # -- wczytujemy plik do pamięci --
        fle = open(self.fname, 'r+')
        a = fle.readlines() # zapisuje linie pliku w tablicy
        fle.close() # zamykamy obiekt pliku, nie będzie nam więcej potrzebny

        # -- czytamy dalej --
        # nazwa źródła
        self.sourcename = a[0].split("\'")[1].strip()
        self.INT = float( (a[0].split())[1])

        # rektascensja
        tmp = a[1].split()
        self.RA = float(tmp[1]) + float(tmp[2]) / 60.0 + float(tmp[3]) / 3600.0
        self.rah = int(tmp[1])
        self.ram = int(tmp[2])
        self.ras = int(tmp[3])
        # deklinacja
        if float(tmp[4]) > 0:
            self.DEC = float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0
        else:
            self.DEC = -1.0 * (-1.0 * float(tmp[4]) + float(tmp[5]) / 60.0 + float(tmp[6]) / 3600.0)

        self.decd = int(tmp[4])
        self.decm = int(tmp[5])
        self.decs = int(tmp[6])
        # epoka
        self.epoch = float(tmp[7])

        # azymut i elewacja
        tmp = a[2].split()
        self.AZ = float(tmp[1]) # azymut
        self.EL = float(tmp[2]) # elewacja
        self.azd = int(self.AZ)
        self.azm = int(60 * (self.AZ % 1))
        self.eld = int(self.EL)
        self.elm = int(60 * (self.EL % 1))
        # czas
        tmp = a[4].split()
        # UT
        self.UTh = float(tmp[1]) # godzina UT
        self.UTm = float(tmp[2]) # minuta UT
        self.UTs = float(tmp[3]) # sekunda UT
        # ST
        self.STh = int(tmp[4]) # godzina ST
        self.STm = int(tmp[5]) # minuta ST
        self.STs = int(tmp[6]) # sekunda ST
        # data
        tmp = a[5].split()
        self.lsec = float(tmp[1]) # sekunda linukskowa
        self.dayn = float(tmp[2]) # dzień roku
        self.year = float(tmp[7]) # rok
        # szukamy miesiąca
        self.monthname = tmp[4]
        self.monthtab = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        self.month = float(self.monthtab.index(self.monthname)) + 1
        self.day = float(tmp[5])
        
        # -- reszta dat - przeliczamy je --
        # na początek konstruujemy formatkę: YYYY-MM-DDTHH:MM:SS 
        self.isotime = str(int(self.year)) + "-"# + str(int(self.month))

        if len(str(int(self.month))) == 1:
            self.isotime = self.isotime + "0" + str(int(self.month)) + "-"
        else:
            self.isotime = self.isotime + str(int(self.month)) + "-"

        if len(str(int(self.day))) == 1:
            self.isotime = self.isotime + "0" + str(int(self.day)) + "T"
        else:
            self.isotime = self.isotime + str(int(self.day)) + "T"

        if len(str(int(self.UTh))) == 1:
            self.isotime = self.isotime + "0" + str(int(self.UTh)) + ":"
        else:
            self.isotime = self.isotime + str(int(self.UTh)) + ":"

        if len(str(int(self.UTm))) == 1:
            self.isotime = self.isotime + "0" + str(int(self.UTm)) + ":"
        else:
            self.isotime = self.isotime + str(int(self.UTm)) + ":"

        if len(str(int(self.UTs))) == 1:
            self.isotime = self.isotime + "0" + str(int(self.UTs))
        else:
            self.isotime = self.isotime + str(int(self.UTs))

        # nasza formatka nazywa się "isotime" i służy jako argument 
        # do funkcji "Time" z pakietu astropy.time
        # -- oblczamy czasy --
        self.tee = Time(self.isotime, format="isot", scale="utc")
        self.decimalyear = self.tee.decimalyear
        self.jd = self.tee.jd
        self.mjd = self.tee.mjd

        # -- tworzymy stringi do zapisu w pliku --
        # tym razem chodzi o coś w stylu 270421
        self.datestring = ""
        if len(str(int(self.day))) == 1:
            self.datestring = self.datestring + "0" + str(int(self.day))
        else:
            self.datestring = self.datestring + str(int(self.day))

        if len(str(int(self.month))) == 1:
            self.datestring = self.datestring + "0" + str(int(self.month))
        else:
            self.datestring = self.datestring + str(int(self.month))
        
        self.datestring = self.datestring + str(int(self.year - 2000))

        # -- częstotliwości --
        self.freq = []
        self.freqa = []
        self.rest = []
        self.bbcfr = []
        self.bbcnr = []
        self.polnames = []
        self.bw = []
        self.vlsr = []
        self.lo = []
        self.tsys = []

        tmp = a[6].split()
        for i in range(len(tmp)-1):
            self.freq.append(float(tmp[i+1]))
        
        tmp = a[7].split()
        for i in range(len(tmp)-1):
            self.freqa.append(float(tmp[i+1]))

        tmp = a[8].split()
        for i in range(len(tmp)-1):
            self.rest.append(float(tmp[i+1]))

        tmp = a[9].split()
        for i in range(len(tmp)-1):
            self.bbcfr.append(float(tmp[i+1]))

        tmp = a[10].split()
        for i in range(len(tmp)-1):
            self.bbcnr.append(int(tmp[i+1]))

        tmp = a[11].split()
        for i in range(len(tmp)-1):
            self.bw.append(float(tmp[i+1]))

        tmp = a[12].split()
        for i in range(len(tmp)-1):
            self.polnames.append(tmp[i+1])

        tmp = a[13].split()
        for i in range(len(tmp)-1):
            self.vlsr.append(float(tmp[i+1]))

        tmp = a[14].split()
        for i in range(len(tmp)-1):
            self.lo.append(float(tmp[i+1]))

        tmp = a[15].split()
        for i in range(len(tmp)-1):
            self.tsys.append(float(tmp[i+1]))
        
        self.freq = asarray(self.freq)
        self.freqa = asarray(self.freqa)
        self.rest = asarray(self.rest)
        self.bbcfr = asarray(self.bbcfr)
        self.bbcnr = asarray(self.bbcnr)
        self.polnames = asarray(self.polnames)
        self.bw = asarray(self.bw)
        self.vlsr = asarray(self.vlsr)
        self.lo = asarray(self.lo)
        self.tsys = asarray(self.tsys)

        self.read_data(a[19:])

    def read_data(self, a):
        # -- deklarujemy kontenery dla konkretnych BBC --
        self.auto = []
        self.bbc1I = []
        self.bbc2I = []
        self.bbc3I = []
        self.bbc4I = []
        self.no_of_channels = 4097
        # -- zapełniamy BBC --
        for i in range(self.no_of_channels):
            tmp = a[i].split()
            self.bbc1I.append(float(tmp[1]))
        for i in range(self.no_of_channels, 2*self.no_of_channels):
            tmp = a[i].split()
            self.bbc2I.append(float(tmp[1]))
        for i in range(2*self.no_of_channels, 3*self.no_of_channels):
            tmp = a[i].split()
            self.bbc3I.append(float(tmp[1]))
        for i in range(3*self.no_of_channels, 4*self.no_of_channels):
            tmp = a[i].split()
            self.bbc4I.append(float(tmp[1]))
        
        # -- agregujemy do jednej tablicy --
        self.auto.append(self.bbc1I)
        self.auto.append(self.bbc2I)
        self.auto.append(self.bbc3I)
        self.auto.append(self.bbc4I)

        # -- zamieniamy na numpy array --
        self.auto = asarray(self.auto)
        # ----- Koniec czytania danych --

    def correct_auto(self, scannr = 1):
        # -- obliczamy średnią z ostatnich 240 kanałów --
        self.average = []
        for i in range(len(self.auto)):
            if mean(self.auto[i][3857:]) == 0.0:
                self.average.append(1.0)
                print("-------> BBC", i, "corrupted!")
            else:
                self.average.append(mean(self.auto[i][3857:]))
        self.average = asarray(self.average)

        # -- generujemy tablicę z wartościami z pierwszych pikseli --
        # które notabene nie są kanałami pomiarowymi i do tego nie służą
        self.auto0tab = []
        for i in range(len(self.auto)):
            self.auto0tab.append(self.auto[i][0])
        self.auto0tab = asarray(self.auto0tab)
        # -- liczymy prawdziwą ilość "samples accumulated" - cokolwiek to znaczy --
        # -- liczymy multiple --
        self.multiple = []
        for i in range(len(self.auto)):
                self.multiple.append(int(nint(self.auto0tab[i] / self.average[i])))
        self.multiple = asarray(self.multiple)

        # -- liczymy Nmax --
        self.Nmax = []
        for i in range(len(self.auto)):
            self.Nmax.append(int(self.auto0tab[i] / self.multiple[i]))

        # -- liczymy bias --
        self.bias0 = self.average / self.Nmax - 1

        # -- edytujemy dane, pozbawiając się intencjonalnego biasu --
        for i in range(len(self.auto)):
            self.auto[i] = self.auto[i] - self.Nmax[i]

        # -- tworzymy zabawę, która się nazywa "zero lag autocorrelation" --
        # -- jest to poprostu wartość funkcji autokorelacji dla zapóźnienia, równego 0 --
        self.zero_lag_auto = []
        for i in range(len(self.auto)):
            self.zero_lag_auto.append(self.auto[i][1])
        self.zero_lag_auto = asarray(self.zero_lag_auto)

        # -- to samo, tylko znormalizowane --
        self.r0 = self.zero_lag_auto / self.Nmax

        # -- normalizujemy całą funkcję autokorelacji --
        for i in range(len(self.auto)):
            self.auto[i] = self.auto[i] / self.zero_lag_auto[i]

        # -- korekta ze względu na kwantyzację sygnału --
        for i in range(len(self.auto)):
            for j in range(len(self.auto[i])):
                self.auto[i][j] = correctACF(self.auto[i][j], self.r0[i], self.bias0[i])

    def hanning_smooth(self):
        # wygładzamy funkcję autokorelacji
        for i in range(len(self.auto)):
            for j in range(1,len(self.auto[i]), 1):
                cosine = cos(pi * (j-1) / self.NN )**2.0
                self.auto[i][j] = self.auto[i][j] * cosine
    
    def doppset(self, source_JNOW_RA, source_JNOW_DEC, szer_geog, dl_geog, height):
        # -------------- PRĘDKOŚCI --------------------
        # -- liczymy prędkość wokół barycentrum + rotacja wokół własnej osi --
        # rzutowane na źródło
        self.baryvel, hjd = helcorr(obs_long = dl_geog, obs_lat = szer_geog, obs_alt = height, ra2000 = self.RA*15, dec2000 = self.DEC, jd=self.tee.jd)
        # -- liczymy prędkość w lokalnym standardzie odniesienia --
        # rzutowane na źródło
        self.lsrvel = lsr_motion(source_JNOW_RA, source_JNOW_DEC, self.decimalyear)

        # -- prędkość dopplerowska to będzie ich suma --
        self.Vdop = self.baryvel + self.lsrvel
        # ----------------------------------------------

        # --------------- ROTACJA WIDMA ----------------
        # --- rotowanie oryginalnego widma ---
        # przesuwamy linię na 1/4 wstęgi
        self.lo[0] = self.lo[0] - (self.bw[0] / 4)
        # faktyczna częstotliwość obserwowana
        self.fsky = self.rest - self.rest * (-self.Vdop + self.vlsr) / self.c
        # częstotliwość bbc linii widmowej
        self.f_IF = self.fsky - self.lo[0]
        # częstotliwość video - nie jestem pewien dokładnie po co to 
        self.fvideo = []
        for i in range(len(self.auto)):
            self.fvideo.append(self.f_IF[i] - copysign(self.bbcfr[i], self.f_IF[i]))
        self.fvideo = asarray(self.fvideo)

        # kanał, w którym jest linia w domenie częstotliwości
        # F DUCT
        self.NNch = len(self.auto[0]) - 1 # nnch jest rodzaju INT
        self.kanalf = []
        for i in range(len(self.auto)):
            self.kanalf.append(int(self.NNch * abs(self.fvideo[i]) / self.bw[i] + 1))
        self.kanalf = asarray(self.kanalf)

        # wprowadzamy Q
        # Q* ilość kanałów = nr.kanału, w którym leży linia
        self.q = []
        self.kanalv = []
        for i in range(len(self.auto)):
            if self.fvideo[i] < 0.0:
                self.kanalf[i] = self.NNch - self.kanalf[i] + 1
                self.q.append(-self.fvideo[i] / self.bw[i])
            else:
                self.q.append(1.0 - self.fvideo[i] / self.bw[i] - 1.0 / self.NNch)
        self.q = asarray(self.q)
        # robimy kanalv
        self.kanalv = self.NNch - self.kanalf + 1

        # prędkość w kanale 1024 w spoektrum częstotliwości
        self.v1024f = self.vlsr + (1024 - self.kanalf) * (-self.c * self.bw) / (self.rest * self.NNch)
        # prędkość w kanale 1024 w spektrum prędkości
        self.v1024v = self.vlsr + (1024 - self.kanalv) * (-self.c * self.bw) / (self.rest * self.NNch)
        
        # ilość kanałów, o które trzeba przerotować widmo 
        self.fc = self.q * self.NNch - 1024
        self.fcBBC = self.fc
        
        # -- przygotowujemy dane do fft --
        self.fr = self.fc * 2.0 * pi / self.NN 

        # -- dwa indeksy --
        # tutaj mamy coś takiego:
        self.auto_prepared_to_fft = []
        for i in range(len(self.auto)):
            self.auto_prepared_to_fft.append(zeros(2 * self.NN + 1))
        self.auto_prepared_to_fft = asarray(self.auto_prepared_to_fft)

        # -- tworzymy tablice --
        self.auto_prepared_to_fft = zeros((4, self.NN), dtype=complex128) # docelowa
        # -- przygotowujemy funkcje autokorelacji do FFT --
        for w in range(len(self.auto)): # iteruje po bbc
            # generujemy tablice tymczasowe
            self.auto_prepared_to_fft_real = zeros((self.NN)) # tymczasowa, rzeczywiste
            self.auto_prepared_to_fft_imag = zeros((self.NN)) # tymczasowa, urojone
            # zapełniamy je
            for i in range(0, int(self.NN / 2)): # iteruje po kanałach bbc
                # -- fazy do rotacji widma --
                sin_phase = sin( (i) * self.fr[w])
                cos_phase = cos( (i) * self.fr[w])
                
                self.auto_prepared_to_fft_real[i] = self.auto[w][i+1] * cos_phase # część rzeczywista
                self.auto_prepared_to_fft_imag[i] = self.auto[w][i+1] * sin_phase # część zespolona
                self.auto_prepared_to_fft_real[-i] = self.auto_prepared_to_fft_real[i] # mirror części rzeczywistej
                self.auto_prepared_to_fft_imag[-i] = -self.auto_prepared_to_fft_imag[i] # mirror części zespolonej
                
            # korzystając z wektoryzacji łączymy tablice
            self.auto_prepared_to_fft_real[int(self.NN / 2)] = 0.0
            self.auto_prepared_to_fft_imag[int(self.NN / 2)] = 0.0
            #self.auto_prepared_to_fft_real[0] = 0.0
            #self.auto_prepared_to_fft_imag[0] = 0.0
            # ustawiamy jeszcze początek
            self.auto_prepared_to_fft[w].real = self.auto_prepared_to_fft_real
            self.auto_prepared_to_fft[w].imag = self.auto_prepared_to_fft_imag
        # --------------

    # -- liczy kilka parametrów --
    def do_statistics(self):
        # liczymy rmean
        self.rMean = self.bias0 * 100.0

        # liczymy ACF0
        self.ACF0 = self.r0

        # liczymy niepewności
        self.V_sigma = []
        self.V_sigma.append(ClipLevel(self.r0[0]))
        self.V_sigma.append(ClipLevel(self.r0[1]))
        self.V_sigma.append(ClipLevel(self.r0[2]))
        self.V_sigma.append(ClipLevel(self.r0[3]))
        self.V_sigma = asarray(self.V_sigma)
    
    # -- skaluje widmo w mili kelwinach --
    def scale_tsys_to_mK(self):
        # pętla po 4 bbc
        for i in range(len(self.auto)):
            self.tsys[i] = self.tsys[i] * 1000.0
            if self.tsys[i] < 0.0:
                self.tsys[i] = 1000.0 * 1000.0

    def make_transformata_furiata(self):
        # -- ekstrahujemy odpowiednie tablice --
        self.tab_to_fft = []
        for i in range(len(self.auto_prepared_to_fft)):
            self.tab_to_fft.append(self.auto_prepared_to_fft[i])
        self.tab_to_fft = asarray(self.tab_to_fft)
        # -- wykonujemy transformatę furiata --
        self.spectr_bbc = []
        for i in range(len(self.auto)):
            self.spectr_bbc.append(fft(self.tab_to_fft[i]).real)
        self.spectr_bbc = asarray(self.spectr_bbc)

        # -- ekstra*ujemy odpowiednie części --
        self.spectr_bbc_final = []
        for i in range(len(self.auto)):
            # -- sprawdzamy, czy berzemy dolne czy górne:
            if self.fvideo[i] > 0: # jak tak, to górne
                self.spectr_bbc_final.append(self.spectr_bbc[i][int(self.NN / 2):])
            else: # jak nie, to dolne
                self.spectr_bbc_final.append(self.spectr_bbc[i][:int(self.NN / 2)])
        self.spectr_bbc_final = asarray(self.spectr_bbc_final)

    # -- kalibruje dane w tsys --
    def calibrate_in_tsys(self):
        for i in range(len(self.auto)):
            self.spectr_bbc_final[i] = self.spectr_bbc_final[i] * self.tsys[i]

    # -- wyświetla rozszerzone informacje o procedurze --
    def extended_print(self):
        print('f(LSR)/MHz   f(sky)      LO1(RF)    LO2(BBC)   fvideo   v(Dopp) [km/s] V(LSR)')
        #print(tab[i].rest[0], tab[i].fsky[0], tab[i].lo[0], tab[i].bbcfr[0], tab[i].fvideo[0], -Vdop, tab[i].vlsr)
        print('%.3f    %.3f    %.3f    %.3f    %.3f    %.3f       %.3f' % (self.rest[0], self.fsky[0], self.lo[0], self.bbcfr[0], self.fvideo[0], -self.Vdop, self.vlsr[0]))
        print('====> Frequency domain: line is in', self.kanalf[0], '=', round(self.v1024f[0],3), 'km/s')
        print('====> Velocity domain: line is in', self.kanalv[0], '=', round(self.v1024v[0],3), 'km/s')
        print('Output spectra were rotated by', round(self.fcBBC[0],3), 'channels')
        if self.fvideo[0] > 0:
            date6 = ' (USBeff)'
        else:
            date6 = ' (LSBeff)'
        print('ACFs', date6, 'Nmax =', int(self.Nmax[3]), '    BBC1   ', '  BBC2   ', '   BBC3   ', '  BBC4')
        print("r0 =                                %.4f    %.4f    %.4f    %.4f" % (self.ACF0[0], self.ACF0[1], self.ACF0[2], self.ACF0[3]))
        print("rmean (bias of 0) =                 %.4f    %.4f    %.4f    %.4f" % (self.rMean[0], self.rMean[1], self.rMean[2], self.rMean[3]))
        print("Threshold (u=V/rms) =               %.4f    %.4f    %.4f    %.4f" % (self.V_sigma[0], self.V_sigma[1], self.V_sigma[2], self.V_sigma[3]))
    
# ----------------------------------


# ---- powiadomienie powitalne ----
print("-----------------------------------------")
print("-----> Welcome to A3S")
print("-----> A3S is a tool to make FFT from 4096 channel autocorrelator output")
print("-----> It also shifts line to channel 1024")
if len(argv) < 2:
    print("-----------------------------------------")
    print("-----> WARNING: no list provided!")
    print("-----> USAGE: a3s.py list_of_.DAT_files")
    print("-----> You need to pass list in the argument!")
    print("-----> Exiting...")
    print("-----------------------------------------")
    exit()

# --- metoda wczytująca listę ---
def read_list_of_files(list_filename):
    # -- otwieramy --
    d = open(list_filename, "r+")
    a = d.readlines()
    d.close()
    # ---------------
    # -- czytamy pliki --
    flenames = []
    for i in range(len(a)):
        tmp = a[i].split()
        flenames.append(tmp[0])
    # -------------------
    # -- zwracamy tablicę z nazwami plików --
    return flenames

# ----- zbiór stałych na początek programu -----
c = 2.99792458e+5 # prędkość światła, m/s
dl_geog = 18.56406 # stopni
szer_geog = 53.09546 # stopni
height = 133.61 # wysokość n.p.m.
NN = 8192 # ilość kanałów x 2
# ----------------------------------------------

# -- tablica z klasami --
tab = []    
# -- czytamy listę (weźmie to, co podamy w argumencie programu) --
list_of_files = read_list_of_files(argv[1])
# -- tworzymy klasy --
for i in range(len(list_of_files)):
    tab.append(datfile(list_of_files[i]))
# --------------------

# ---- zarządzanie współrzędnymi ----
# -- deklarujemy obiekt sky coord, w którym będą zawarte współrzędne --
# -- WAŻNE: współrzędne są wzięte z PIERWSZEGO skanu --
source_J2000 = SkyCoord(ra=tab[0].RA*u.hourangle, dec=tab[0].DEC*u.degree, frame=FK5, equinox='J2000')

# ------------- PRECESJA I NUTACJA --------------------
# -- do precesji deklarujemy nowy frame FK5 z epoką pierwszego skanu --
frame_now = FK5(equinox="J" + str(tab[0].decimalyear))
# -- by wykonać precesję i nutację wystarczy teraz: 
source_JNOW = source_J2000.transform_to(frame_now)
# będziemy robić precesję w głównej pętli
# ------------------------------------------------------

# ---------- WSPÓŁRZĘDNE GALAKTYCZNE -------------------
l_ga = source_JNOW.galactic
source_L = (l_ga.l*u.degree).value
source_B = (l_ga.b*u.degree).value
source_ld = int(source_L)
source_lm = int(60.0 * (source_L % 1))
source_bd = int(source_B)
source_bm = int(60.0 * (source_B % 1))
# ------------------------------------------------------

# --- printowanie komunikatu ---
print("-----> Loaded", len(tab), "scans")
print("-----------------------------------------")
# --- zrzynane z A2S kroki, mające na celu doprowadzić nas do końcowego widma ---
for i in range(len(tab)):

    # ------------- PRECESJA I NUTACJA ---------------
    # -- wykonujemy precesję na czas obecnego skanu --
    # -- do precesji deklarujemy nowy frame FK5 z epoką aktualnego skanu --
    frame_now = FK5(equinox="J" + str(tab[i].decimalyear))
    # -- by wykonać precesję i nutację wystarczy teraz: 
    source_JNOW = source_J2000.transform_to(frame_now)
    # -- zapisujemy wartości RA i DEC po precesji do nowych zmiennych --
    source_JNOW_RA = (source_JNOW.ra*u.degree).value
    source_JNOW_DEC = (source_JNOW.dec*u.degree).value
    # ------------------------------------------------

    # -- korekta funkcji autokorelacji --
    # ze względu na 2 i 3 poziomową kwantyzację etc.
    tab[i].correct_auto(scannr = i+1)

    # -- wygładzanie Hanninga --
    tab[i].hanning_smooth()

    # -- korekta na ruch ziemi --
    # obejmuje ona: 
    # 1. ruch wokół własnej osi
    # 2. ruch obiegowy wokół barycentrum US
    # 3. ruch względem lokalnej grupy gwiazd (LSR)
    # dwa pierwsze punkty są zrealizowane z dokładnością do ~ 1 cm/s
    # trzeci opiera się na metodzie, przepisanej żywcem z oryginalnego A2S
    # jej dokładność budzi pewne wątpliwości
    # argumenty doppset: 1: RA po precesji (deg), 2: DEC po precesji (deg)
    # 3: szerokość geograficzna (deg) 4: długość geogradiczna (deg, E > 0, W < 0)
    # 5: wysokość nad geoidą zi-emii
    # doppset wykonuje również rotację f. autokorelacji
    tab[i].doppset(source_JNOW_RA, source_JNOW_DEC, szer_geog, dl_geog, height)
    print("-----> scan %d: line shifted by %4.3f channels" % (i+1, round(tab[i].fcBBC[0],3)))

    # -- kilka statystyk liczymy --
    tab[i].do_statistics()
    
    # -- skalujemy tsys w mK --
    tab[i].scale_tsys_to_mK()

    # --- PRINTOWANIE ---
    # zakomentowane, normalnie tego nie potrzebujemy
    #tab[i].extended_print()

    # -- robimy transformatę fouriera --
    tab[i].make_transformata_furiata()

    # -- kalibrujemy tsys --
    tab[i].calibrate_in_tsys()
print("-----------------------------------------")

# -- zapisujemy --
# plik wynikowy z zapisanymi danymi
print("-----> Saving to file WYNIK.DAT")
fle = open("WYNIK.DAT", "w+")

# pętla zapisująca 
for i in range(len(tab)):
    for ee in range(len(tab[i].auto)):
        # ---- nagłówek ----
        fle.write("???\n")
        fle.write(repr(tab[i].rah).rjust(6) + repr(tab[i].ram).rjust(6) + repr(tab[i].ras).rjust(6) + repr(tab[i].decd).rjust(6) + repr(tab[i].decm).rjust(6) + repr(tab[i].decs).rjust(6) +"\n" )
        fle.write(repr(source_ld).rjust(6) + repr(source_lm).rjust(6) + repr(source_bd).rjust(6) + repr(source_bm).rjust(6) + "\n")
        fle.write(repr(tab[i].azd).rjust(6) + repr(tab[i].azm).rjust(6) + repr(tab[i].eld).rjust(6) + repr(tab[i].elm).rjust(6) + "\n" )
        fle.write(tab[i].datestring.rjust(10) + "\n")
        fle.write(repr(int(tab[i].STh)).rjust(6) + repr(int(tab[i].STm)).rjust(6) + repr(int(tab[i].STs)).rjust(6) + "\n")
        fle.write(repr(round(tab[i].tsys[0] / 1000.0, 3)).rjust(8)  + "\n")
        fle.write("0".rjust(6) + "\n")
        fle.write(repr(ee).rjust(6) + "\n")
        fle.write("$$$\n")
        fle.write(repr(len(tab[i].spectr_bbc_final[ee])).rjust(12) + repr(int(tab[i].bw[ee])).rjust(15) + repr(0.25).rjust(15) + repr(tab[i].vlsr[ee]).rjust(11) + repr(tab[i].rest[ee]).rjust(18) + "\n")
        fle.write(tab[i].sourcename + "\n")
        fle.write("***" + "\n")
        fle.write(repr(int(tab[i].UTh)).rjust(8) + repr(int(tab[i].UTm)).rjust(8) + repr(int(tab[i].UTs)).rjust(8) + repr(int(tab[i].INT)).rjust(8) + "\n")
        # --- dane ----
        for j in range(len(tab[i].spectr_bbc_final[ee])):
            # sprawdzamy, czy to co próbujemy zapisać nie jest aby za długie
            if len(repr( round(tab[i].spectr_bbc_final[ee][j] ,1))) < 9:
                # jeśli tak, zapisujemy
                fle.write( repr( round(tab[i].spectr_bbc_final[ee][j] ,1) ).rjust(10) )
            else:
                # jak nie... to może sp......
                fle.write( repr(000.0).rjust(10) )
            # co 8 wpisów przechodzimy do nowej linii
            if (j + 1) % 8 == 0:
                fle.write("\n")
    # ---------------------------------------------------

# -- zamykamy plik --
fle.close()
print("-----> Completed succesfully. Ending")
print("-----------------------------------------")
