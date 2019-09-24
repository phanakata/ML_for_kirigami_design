15grid_shuffled.dat dataset contains all possible configurations of kirigamis with 0--15 cuts and their corresponding mechanical properties. The dataset was used in <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.255304" style="color:#268cd7
"> **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning, *Phys. Rev. Lett*, 121, 255304  (2018).</a>

Data format:
* Column 0-14: kirigami structure, '1' for no-cut and '0' for cut
* Column 15: yield strain 
* Column 16: "toughness" (integration stress-strain curver up to yield point)
* Column 17: yield stress 

This dataset was generated using AIREBO potential with 1.7 mincutoff (default CH.airebo).
In addition we used a 1.92 mincutoff and we found that the ML still finds the optimal structure within 10 generations. The only difference is that yield strain and yield stress are generally lower for the dataset with 1.92 cutoff.

For more details see <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.069901" style="color:#268cd7
"> Erratum: **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning, *Phys. Rev. Lett*, 123, 069901  (2018).</a>


