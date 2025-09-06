Om bestanden **groter dan 1 GB** goed te ondersteunen, moeten we ervoor zorgen dat de app geen volledige video in het geheugen probeert te laden. In de code is dit grotendeels al goed (we gebruiken `ffmpeg`/`ffprobe` en mappen alleen de telemetrie-stream). Voor uploads en gebruik in Streamlit Cloud moeten we een paar dingen verduidelijken en aanpassen:

1. **Gebruik bestandspad of URL voor >1 GB**  
   Upload via Streamlit werkt slecht voor bestanden groter dan 1 GB (browser + netwerk + RAM). Daarom heb ik in de code al een optie toegevoegd voor:
   - **Bestandspad**: ideaal lokaal.
   - **Publieke URL**: ffmpeg kan direct een remote MP4/MOV lezen, zonder download via de browser.

2. **Voorkom RAM-kopie bij upload**  
   Als iemand tóch uploadt, wordt het bestand tijdelijk volledig in geheugen/buffer geplaatst. Dit is niet geschikt voor >1 GB. Voor zulke bestanden: **niet uploaden, maar bestandspad of URL gebruiken.**

3. **Streamlit Cloud limieten**  
   - Streamlit Cloud heeft standaard een uploadlimiet (~200MB). Deze kan je in `.streamlit/config.toml` verhogen, maar boven 1GB is niet praktisch. Gebruik hier dus altijd een **publieke URL** of werk lokaal.
   - ffmpeg leest dan direct van die URL (HTTP range requests).

4. **Aanpassing in code**  
   Ik heb al `src_type = ["Bestandspad", "Upload", "URL"]` toegevoegd. Daarmee kan de gebruiker grote bestanden via pad of URL aanleveren.

### Hoe je dit praktisch gebruikt
- **Lokaal**: Start de app en kies *Bestandspad*. Geef gewoon `D:/GoPro/clip.MP4` of `/home/user/clip.MP4` in.
- **Cloud**: Upload je video naar een plek waar hij publiek bereikbaar is (bijv. Google Drive gedeeld met link, Dropbox direct link, of eigen webserver). Kies *URL* in de app en plak `https://.../clip.MP4`. ffmpeg haalt alleen de telemetrie-stream op, dus dat gaat snel en gebruikt nauwelijks RAM.

👉 Conclusie: Met de huidige code kun je veilig bestanden >1 GB verwerken zolang je **bestandspad** (lokaal) of **URL** (cloud) gebruikt. Upload in de browser moet je vermijden bij zulke groottes.

Wil je dat ik meteen een helper maak om **automatisch de .gpmd-stream lokaal te extraheren** (klein bestand van enkele MB’s) zodat je alleen dat bestand hoeft te uploaden of mee te geven? Dat is de meest stabiele workflow voor >1 GB video’s.
