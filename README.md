Berikut adalah cara setup chatbot dalam Cloud Run GCP:
1. Buka Command Prompt
2. Ketik gcloud auth login
3. Nanti akan diminta login untuk yang belum pernah login atau muncul google account anda jika sudah pernah login dalam aplikasi web, pilih google account yang memiliki sebuah project dan lanjutkan verifikasi akun
4. Setelah itu, akan muncul tampilan seperti dibawah. Jika project belum sesuai dengan project yang diinginkan, dapat mengetik  gcloud config set project PROJECT_ID. PROJECT_ID dapat diisi dengan project id anda yang dapat dilihat di google cloud.

![Screenshot 2025-06-27 145348](https://github.com/user-attachments/assets/309dc4e2-03fe-4b59-b876-affd8b252802)

5. Setelah itu, pindah ke folder anda menyimpan file chatbot dengan mengetik cd "path location dari foldernya" dalam command prompt
6. Setelah pindah ke folder chatbot, chatbot di deploy ke cloud run dengan mengetik dalam command prompt sebagai berikut:
   gcloud run deploy nama-chatbot --source=. --region=asia-southeast2 --platform=managed --allow-unauthenticated --memory=20Gi --cpu=8 --timeout=900 --max-instances=1 --min-instances=0
Note:
- nama-chatbot dapat diganti sesuai keinginan (harus huruf kecil dan tidak boleh menggunakan "_".
- region dapat disesuaikan dengan kebutuhan (disarankan di asia-southeast2 karena itu server Jakarta,Indonesia).
- Jika memungkinkan untuk upgrade memory dan cpu dapat dibesarkan.
7. Setelah itu tunggu sampai deploy selesai
8. Ketika selesai, chatbot dapat dilihat di GCP bagian Cloud Run
  
![image](https://github.com/user-attachments/assets/1a6a2884-ad5e-4e97-8d76-0d73fe0659cc)
9. Jika sudah masuk ke dalam cloud run, akan terlihat chatbot yang sudah dideploy (dalam contoh gambar berikut, nama chatbot yang saya beri adalah chatbot-aiml 10)
![image](https://github.com/user-attachments/assets/ff6b14e2-bba2-4b53-a88b-1228f89b0c00)
10. Untuk mengakses chatbot, klik nama chatbot dan klik link url yang ada.
![Screenshot 2025-06-27 193352](https://github.com/user-attachments/assets/7c685507-129a-4f4a-8c4f-3435f6ff0ebb)

Berikut adalah cara memasukkan dataset ke dalam bucket di GCS:
1. Dalam halaman utama project, klik garis 3 di kiri atas halaman (di kiri logo google cloud), lalu klik cloud storage > Buckets
2. Jika belum pernah menggunakan bucktes di GCS, maka tampilan akan seperti pada gambar. Untuk membuat bucket baru dapat click create bucket. Jika sudah pernah membuat bucket dan ada bucket lain, dapat klik create disebelah refresh.
![image](https://github.com/user-attachments/assets/79d39ccd-2606-4fa2-80b8-3389de7e1daf)
3. Setelah itu, dapat memberikan nama sesuai keinginan dan mengkonfigurasi region tempat penyimpanan (disarankan menggunakan region yang sama dengan region chatbot)
![Screenshot 2025-06-27 195016](https://github.com/user-attachments/assets/05b78521-02f4-4adc-aaa9-396f90d66a51)
4. Setelah selesai, click create pada bagian akhir halaman dan tampilannya akan seperti gambar berikut
![Screenshot 2025-06-27 195059](https://github.com/user-attachments/assets/64855494-dd03-472a-a195-bd6972bb703f)
5. Untuk menambahkan file, click upload lalu akan muncul 2 pilihan "Upload file" dan "Upload folder". Untuk menambahkan database chatbot, gunakan upload folder. Untuk menambahkan module.json dan question.json, gunakan upload file 
   

