        def D_0_change():
            a.self=1864.84
            b=0
            c=410.1

        def D_0_bar_change():
            a=1864.84
            b=0
            c=410.1

        def D_plus_change():
            self.__a=1869.61
            self.__b=1
            self.__c=1040

        def D_minus_change():
            a=1869.61
            b=-1
            c=1040

        def Ks_change():
            a=497.611
            b=0
            c=89540

        def Kl_change():
            a=497.611
            b=0
            c=5.116*10**7

        def K_plus_change():
            a=493.677
            b=1
            c=1.238*10**7

        def K_minus_change():
            a=493.677
            b=-1
            c=1.238*10**7

        def Pi_0_change():
            a=134.9766
            b=0
            c=0.0852

        def Pi_plus_change():
            a=139.57018
            b=1
            c=2.6033*10**7

        def Pi_minus_change():
            a=139.57018
            b=-1
            c=2.6033*10**7

        def rho_change():
            a=775.26
            b=0
            c=4.5*10**-9

        def rho_plus_change():
            a=775.26
            b=1
            c=4.5*10**-9

        def rho_minus_change():
            a=775.26
            b=-1
            c=4.5*10**-9

        def e_plus_change():
            a=0.510999
            b=1
            c=1*10**100

        def gamma_change():
            a=1*10**-5
            b=0
            c=10**100

        def mu_change():
            a=105.6584
            b=0
            c=2.197*10**9

        def mu__plus_change():
            a=105.6584
            b=1
            c=2.197*10**9

        def mu__minus_change():
            a=105.6584
            b=-1
            c=2.197*10**9

        D_plus = Particle('D+',1864.84,1,410.1)
        D_plus.set_pos([0],[0],[0])
        D_plus.set_v([.01],[.01],[.01])

        D0 = Particle('D0',1869.61,0,1040)
        D0.set_pos([0],[0],[0])
        D0.set_v([.01],[.01],[.01])