<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:27AE60,100:0F5132&height=200&section=header&text=Harare%20Property%20Prices&fontSize=46&fontColor=ffffff&fontAlignY=40&animation=fadeIn" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3500&pause=800&color=27AE60&center=true&vCenter=true&width=680&lines=Predict+the+listing+price+of+any+Harare+house;17+real+suburbs+from+Borrowdale+to+Budiriro;Borehole+%2B+solar+%2B+walled-yard+features" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/harare-property-prices"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

<p align="center">
  <img src="docs/dashboard.png" alt="Dashboard preview" width="900">
</p>


## Why I built this

Most regression tutorials use US housing datasets. I wanted one that reflected **my own market**, so I built a Harare property price model from scratch: 17 real suburbs from Borrowdale and Glen Lorne down to Mufakose and Budiriro, with the features that actually move the needle here — borehole, solar backup, walled yard, plot size, and distance to the CBD.

The dataset is synthetic but the **structure is real**: pricing tiers reflect 2024–2025 USD asking prices, and feature effects (a borehole adds ~5% to the price, solar adds ~4%) are calibrated to what listings actually show.

## At a glance

|  |  |
|---|---|
| **Problem** | Predict the USD listing price of a house in Harare |
| **Suburbs** | Borrowdale · Glen Lorne · Mt Pleasant · Ballantyne Park · Highlands · Vainona · Greendale · Avondale · Marlborough · Avonlea · Belvedere · Hatfield · Mabelreign · Mufakose · Glen View · Kambuzuma · Budiriro |
| **Features** | Suburb · bedrooms · bathrooms · size · plot size · age · borehole · solar · pool · walled |
| **Method** | Stacked ensemble (Random Forest + Gradient Boosting → Ridge meta) |
| **Results** | R² **0.88**, RMSE around **$25k** on the held-out test set |
| **Stack** | scikit-learn · pandas · Streamlit · Plotly |

## How I approached it

1. **Generated 5,000 listings** across 17 suburbs with realistic price tiers and amenity distributions.
2. **EDA** — distributions by suburb, the price gap between premium and township tiers, and which amenities matter most.
3. **Real preprocessing pipeline** — `ColumnTransformer` + `Pipeline` so train/test boundaries are never crossed.
4. **5-fold CV** comparing Linear Regression, Ridge, Random Forest, Gradient Boosting.
5. **Stacked ensemble** — RF + GBR with a Ridge meta-learner.
6. **Interpretation** — permutation importance to see which features really drive price.

## What I found

- **Suburb dominates everything.** Borrowdale is ~10× more expensive than Budiriro at similar floor sizes. Location is doing real work.
- **Borehole adds ~5%, solar adds ~4%** to the median price. With routine load-shedding and water shortages, these aren't luxuries — they're insurance, and buyers pay for them.
- **Distance to CBD is not a clean linear signal**: premium suburbs and townships sit at similar distances. What matters is *which side* of the city, not how far.

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook harare_house_prediction.ipynb
streamlit run dashboard.py
```

The notebook generates `data/harare_listings.csv` on first run — no external download.

## Interactive dashboard

Three tabs:
- **Map of Harare** — every listing plotted at its real latitude/longitude, coloured by price. Filter by tier and the geographic story jumps out.
- **What drives price?** — boxplot of price by suburb (sorted), and the percentage uplift each amenity adds.
- **Price my house** — pick a suburb, set the features, get a live price + comparison to the suburb's median.

## What I'd do next

- Scrape actual listings from Property.co.zw and re-train on real data.
- Add proximity to schools, shopping centres, arterial roads.
- Build a rental yield model alongside the sale model — equally useful for Zim investors.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:27AE60,100:0F5132&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>
