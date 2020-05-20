from gpsearch import get_cases, plot_error


def main():

    cases = [ ("e_lhs", "LHS"), 
              ("e_US", "US"),
              ("e_IVR", "IVR"),
              ("e_US_LW", "US-LW"),
              ("e_IVR_LW", "IVR-LW") ]

    err_list, labels = get_cases(cases)

    plot_error(err_list, 
               logscale=True, 
               filename="test.pdf", 
               labels=labels, 
               cmap="cbrewer2", 
               sig=0.5, 
               xticks=[0,30,60])


if __name__ == "__main__":
    main()


