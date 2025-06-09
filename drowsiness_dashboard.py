import streamlit as st
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
from threading import Thread
import queue
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import logging
from datetime import datetime
import pandas as pd
import requests
import io
import base64
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_notification_time = {}
        self.notification_cooldown = 30  # Cooldown 30 detik untuk mencegah spam
        
    def send_message(self, message):
        """Kirim pesan teks ke Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return None
    
    def send_photo(self, image_array, caption=""):
        """Kirim foto dengan caption ke Telegram"""
        try:
            # Konversi numpy array ke bytes
            img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            url = f"{self.base_url}/sendPhoto"
            files = {'photo': ('alert.jpg', img_bytes, 'image/jpeg')}
            data = {
                'chat_id': self.chat_id,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, files=files, data=data, timeout=15)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return None
    
    def can_send_notification(self, alert_type):
        """Cek apakah bisa mengirim notifikasi (cooldown)"""
        current_time = time.time()
        last_time = self.last_notification_time.get(alert_type, 0)
        
        if current_time - last_time >= self.notification_cooldown:
            self.last_notification_time[alert_type] = current_time
            return True
        return False
    
    def send_alert(self, alert_type, frame=None, metrics=None):
        """Kirim alert ke Telegram dengan foto"""
        if not self.can_send_notification(alert_type):
            return
        
        current_time = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        
        if alert_type == "drowsy":
            message = f"🚨 <b>PERINGATAN DROWSINESS</b> 🚨\n\n"
            message += f"⏰ Waktu: {current_time}\n"
            message += f"👁️ EAR: {metrics.get('ear', 0):.3f}\n"
            message += f"📊 Status: Sopir terdeteksi mengantuk\n"
            message += f"🚗 Segera istirahat atau ganti sopir!"
            
        elif alert_type == "yawn":
            message = f"🔴 <b>BAHAYA KRITIS</b> 🔴\n\n"
            message += f"⏰ Waktu: {current_time}\n"
            message += f"👁️ EAR: {metrics.get('ear', 0):.3f}\n"
            message += f"👄 Lip Distance: {metrics.get('lip_distance', 0):.1f}\n"
            message += f"📊 Status: Sopir mengantuk + menguap\n"
            message += f"🛑 HENTIKAN KENDARAAN SEGERA!"
        
        # Kirim foto dengan caption jika ada frame
        if frame is not None:
            Thread(target=self.send_photo, args=(frame, message), daemon=True).start()
        else:
            Thread(target=self.send_message, args=(message,), daemon=True).start()
    
    def send_session_report(self, detector):
        """Kirim laporan sesi"""
        session_duration = datetime.now() - detector.session_start
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        message = f"📊 <b>LAPORAN SESI MENGEMUDI</b> 📊\n\n"
        message += f"⏰ Durasi: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n"
        message += f"😴 Total Drowsy Alerts: {detector.drowsy_count}\n"
        message += f"🥱 Total Yawn Alerts: {detector.yawn_count}\n"
        message += f"🚨 Total Alerts: {detector.drowsy_count + detector.yawn_count}\n\n"
        
        if detector.drowsy_count + detector.yawn_count == 0:
            message += "✅ Sesi berjalan dengan aman!"
        else:
            message += "⚠️ Perhatikan pola mengemudi dan istirahat yang cukup!"
        
        Thread(target=self.send_message, args=(message,), daemon=True).start()

class DrowsinessDetector:
    def __init__(self):
        # Konstanta
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 30
        self.YAWN_THRESH = 20
        self.BUZZER_PIN = 21
        
        # Status alarm
        self.alarm_status = False
        self.alarm_status2 = False
        self.COUNTER = 0
        self.lcd_showing_alert = False
        
        # Statistik
        self.drowsy_count = 0
        self.yawn_count = 0
        self.session_start = datetime.now()
        self.alerts_log = []
        
        # Telegram Bot (akan diinisialisasi di UI)
        self.telegram_bot = None
        
        # Inisialisasi hardware
        self.init_hardware()
        
        # Load model
        self.init_models()
        
        # Queue untuk frame
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.lcd_state = 0  # untuk menentukan tampilan yang sedang ditampilkan
        self.last_lcd_update = time.time()
        
    def init_hardware(self):
        """Inisialisasi LCD dan GPIO"""
        try:
            self.lcd = CharLCD('PCF8574', 0x27)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
            GPIO.output(self.BUZZER_PIN, GPIO.LOW)
            logger.info("Hardware initialized successfully")
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            self.lcd = None
    
    def init_models(self):
        """Load detector dan predictor"""
        try:
            self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            st.error("Gagal memuat model deteksi. Pastikan file model tersedia.")
    
    def set_telegram_bot(self, bot_token, chat_id):
        """Set Telegram Bot"""
        if bot_token and chat_id:
            self.telegram_bot = TelegramBot(bot_token, chat_id)
            return True
        return False
    
    def eye_aspect_ratio(self, eye):
        """Hitung Eye Aspect Ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def final_ear(self, shape):
        """Hitung EAR untuk kedua mata"""
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)
    
    def lip_distance(self, shape):
        """Hitung jarak bibir untuk deteksi menguap"""
        top_lip = np.concatenate((shape[50:53], shape[61:64]))
        low_lip = np.concatenate((shape[56:59], shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance
    
    def trigger_alarm(self, alert_type, frame=None, metrics=None):
        """Trigger alarm, LCD, dan Telegram"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if alert_type == "drowsy":
            self.drowsy_count += 1
            self.alerts_log.append({"time": current_time, "type": "Drowsiness", "severity": "High"})
        elif alert_type == "yawn":
            self.yawn_count += 1
            self.alerts_log.append({"time": current_time, "type": "Yawn + Drowsy", "severity": "Critical"})
        
        # Update LCD
        if self.lcd and not self.lcd_showing_alert:
            try:
                self.lcd.clear()
                self.lcd.write_string("Warning, Driver      Ngantuk!!!")
                self.lcd_showing_alert = True
            except Exception as e:
                logger.error(f"LCD error: {e}")
        
        # Kirim notifikasi Telegram
        if self.telegram_bot:
            self.telegram_bot.send_alert(alert_type, frame, metrics)
        
        # Buzzer alarm (non-blocking)
        Thread(target=self._buzzer_alarm, daemon=True).start()
    
    def _buzzer_alarm(self):
        """Buzzer alarm dalam thread terpisah"""
        try:
            for _ in range(3):  # Buzz 3 kali
                if self.alarm_status or self.alarm_status2:
                    GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(self.BUZZER_PIN, GPIO.LOW)
                    time.sleep(0.5)
        except Exception as e:
            logger.error(f"Buzzer error: {e}")
            
    def update_lcd(self, metrics):
        """Memperbarui tampilan LCD berdasarkan rotasi"""
        try:
            # Cek apakah waktu sudah cukup untuk mengganti tampilan
            if time.time() - self.last_lcd_update > 3:  # Ganti tampilan setiap 3 detik
                self.last_lcd_update = time.time()
                if self.lcd_state == 0:
                    # Status + Waktu
                    current_time = datetime.now().strftime("%H:%M")
                    self.lcd.clear()
                    self.lcd.write_string(f"System: READY")
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(f"{current_time} NORMAL")
                elif self.lcd_state == 1:
                    # Statistik Sesi
                    session_duration = datetime.now() - self.session_start
                    hours, remainder = divmod(session_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    alerts = self.drowsy_count + self.yawn_count
                    self.lcd.clear()
                    self.lcd.write_string(f"Alerts: {alerts} {int(hours):02d}:{int(minutes):02d}h")
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(f"Status: NORMAL")
                elif self.lcd_state == 2:
                    # EAR Value Real-time
                    ear_value = metrics['ear'] # Menggunakan nilai EAR terbaru
                    self.lcd.clear()
                    self.lcd.write_string(f"EAR: {ear_value:.3f}")
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(f"Driver: {'ALERT' if self.alarm_status else 'NORMAL'}")
                
                # Rotasi ke tampilan berikutnya
                self.lcd_state = (self.lcd_state + 1) % 3
        except Exception as e:
            logger.error(f"LCD update error: {e}")

    
    def process_frame(self, frame):
        """Proses frame untuk deteksi drowsiness"""
        if frame is None:
            return None, {}
        
        # Resize untuk performa lebih baik
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        rects = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        metrics = {
            "ear": 0.0,
            "lip_distance": 0.0,
            "drowsy_alert": False,
            "yawn_alert": False,
            "face_detected": len(rects) > 0
        }
        
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Hitung EAR dan lip distance
            ear, leftEye, rightEye = self.final_ear(shape)
            distance = self.lip_distance(shape)
            
            metrics["ear"] = ear
            metrics["lip_distance"] = distance
            
            # Gambar kontur mata dan bibir
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            lip = shape[48:60]
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 2)
            
            # Deteksi drowsiness
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.alarm_status:
                        self.alarm_status = True
                        # Kirim frame saat trigger alarm
                        self.trigger_alarm("drowsy", frame.copy(), metrics)
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    metrics["drowsy_alert"] = True
            else:
                self.COUNTER = 0
                self.alarm_status = False
                if self.lcd_showing_alert:
                    try:
                        self.lcd.clear()
                        self.lcd_showing_alert = False
                    except:
                        pass
            
            # Deteksi yawn + drowsy
            if distance > self.YAWN_THRESH and ear < self.EYE_AR_THRESH:
                cv2.putText(frame, "YAWN + DROWSY ALERT!", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not self.alarm_status2:
                    self.alarm_status2 = True
                    # Kirim frame saat trigger alarm kritis
                    self.trigger_alarm("yawn", frame.copy(), metrics)
                metrics["yawn_alert"] = True
            else:
                self.alarm_status2 = False
            
            # Tampilkan metrics pada frame
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"LIP: {distance:.1f}", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status indicator
            status_color = (0, 255, 0)  # Hijau = normal
            status_text = "NORMAL"
            
            if metrics["yawn_alert"]:
                status_color = (0, 0, 255)  # Merah = critical
                status_text = "CRITICAL"
            elif metrics["drowsy_alert"]:
                status_color = (0, 165, 255)  # Orange = warning
                status_text = "WARNING"
            
            cv2.putText(frame, status_text, (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            break  # Proses hanya wajah pertama yang terdeteksi
        
        self.update_lcd(metrics)
        
        return frame, metrics

# Inisialisasi detector
@st.cache_resource
def get_detector():
    return DrowsinessDetector()

def main():
    st.title("🚗 Driver Drowsiness Detection System with Telegram Bot")
    st.markdown("---")
    
    # Sidebar untuk kontrol
    st.sidebar.title("⚙️ Kontrol Sistem")
    
    # Konfigurasi Telegram Bot
    st.sidebar.subheader("📱 Telegram Bot Configuration")
    bot_token = st.sidebar.text_input(
        "Bot Token", 
        type="password", 
        help="Dapatkan dari @BotFather di Telegram"
    )
    chat_id = st.sidebar.text_input(
        "Chat ID", 
        help="ID chat/group untuk menerima notifikasi"
    )
    
    # Test Telegram connection
    test_telegram = st.sidebar.button("🧪 Test Telegram Bot")
    
    detector = get_detector()
    
    # Set up Telegram bot jika sudah dikonfigurasi
    telegram_configured = False
    if bot_token and chat_id:
        telegram_configured = detector.set_telegram_bot(bot_token, chat_id)
        st.sidebar.success("✅ Telegram Bot dikonfigurasi")
    else:
        st.sidebar.warning("⚠️ Telegram Bot belum dikonfigurasi")
    
    if test_telegram and telegram_configured:
        test_message = "🧪 Test notifikasi dari Driver Drowsiness Detection System\n\n✅ Koneksi berhasil!"
        result = detector.telegram_bot.send_message(test_message)
        if result and result.get('ok'):
            st.sidebar.success("✅ Test berhasil! Cek Telegram Anda.")
        else:
            st.sidebar.error("❌ Test gagal. Periksa Bot Token dan Chat ID.")
    
    # Parameter konfigurasi
    st.sidebar.subheader("Konfigurasi Deteksi")
    detector.EYE_AR_THRESH = st.sidebar.slider(
        "Eye Aspect Ratio Threshold", 0.1, 0.5, 0.2, 0.01
    )
    detector.EYE_AR_CONSEC_FRAMES = st.sidebar.slider(
        "Consecutive Frames", 10, 60, 30, 5
    )
    detector.YAWN_THRESH = st.sidebar.slider(
        "Yawn Threshold", 10, 40, 20, 1
    )
    
    # Kontrol kamera
    st.sidebar.subheader("📹 Kamera")
    camera_index = st.sidebar.selectbox("Pilih Kamera", [0, 1, 2])
    start_detection = st.sidebar.button("🎥 Mulai Deteksi")
    stop_detection = st.sidebar.button("⏹️ Stop Deteksi")
    send_report = st.sidebar.button("📊 Kirim Laporan ke Telegram")
    
    # Kirim laporan sesi
    if send_report and telegram_configured:
        detector.telegram_bot.send_session_report(detector)
        st.sidebar.success("📊 Laporan dikirim ke Telegram!")
    
    # Layout utama
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Metrics Real-time")
        metrics_placeholder = st.empty()
        
        st.subheader("🚨 Status Alert")
        alert_placeholder = st.empty()
        
        st.subheader("📈 Statistik Sesi")
        stats_placeholder = st.empty()
        
        # Status Telegram
        st.subheader("📱 Status Telegram")
        if telegram_configured:
            st.success("✅ Telegram Bot Aktif")
        else:
            st.error("❌ Telegram Bot Tidak Aktif")
    
    # Log alerts
    st.subheader("📋 Log Alerts")
    log_placeholder = st.empty()
    
    # Session state untuk kontrol
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    if start_detection:
        st.session_state.detection_active = True
        if telegram_configured:
            start_message = "🚗 <b>SISTEM MONITORING DIMULAI</b>\n\n"
            start_message += f"⏰ Waktu Mulai: {datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}\n"
            start_message += "📱 Notifikasi aktif untuk peringatan drowsiness"
            detector.telegram_bot.send_message(start_message)
    
    if stop_detection:
        st.session_state.detection_active = False
        if telegram_configured:
            detector.telegram_bot.send_session_report(detector)
    
    # Main detection loop
    if st.session_state.detection_active:
        # Inisialisasi kamera dengan buffer kecil untuk mengurangi lag
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS untuk stabilitas
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Gagal membuka kamera. Periksa koneksi kamera.")
            return
        
        try:
            while st.session_state.detection_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("Gagal membaca frame dari kamera")
                    break
                
                # Clear buffer untuk mengurangi lag
                cap.grab()
                
                # Proses frame
                processed_frame, metrics = detector.process_frame(frame)
                
                if processed_frame is not None:
                    # Konversi ke RGB untuk Streamlit
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update video feed
                    video_placeholder.image(
                        frame_rgb, 
                        channels="RGB", 
                        use_container_width=True
                    )
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.metric("EAR", f"{metrics['ear']:.3f}")
                            st.metric("Lip Distance", f"{metrics['lip_distance']:.1f}")
                        
                        with col_m2:
                            face_status = "✅ Detected" if metrics['face_detected'] else "❌ Not Detected"
                            st.metric("Face", face_status)
                            
                            # Status keseluruhan
                            if metrics['yawn_alert']:
                                st.error("🚨 CRITICAL: Yawn + Drowsy!")
                            elif metrics['drowsy_alert']:
                                st.warning("⚠️ WARNING: Drowsiness!")
                            else:
                                st.success("✅ Driver Alert")
                    
                    # Update alert status
                    with alert_placeholder.container():
                        if metrics['yawn_alert']:
                            st.error("🚨 BAHAYA: Sopir mengantuk dan menguap!")
                        elif metrics['drowsy_alert']:
                            st.warning("⚠️ PERINGATAN: Sopir terlihat mengantuk!")
                        else:
                            st.success("✅ Sopir dalam kondisi siaga")
                    
                    # Update statistik
                    session_duration = datetime.now() - detector.session_start
                    hours, remainder = divmod(session_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    
                    with stats_placeholder.container():
                        st.metric("Durasi Sesi", f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
                        st.metric("Total Drowsy Alerts", detector.drowsy_count)
                        st.metric("Total Yawn Alerts", detector.yawn_count)
                    
                    # Update log
                    if detector.alerts_log:
                        df_log = pd.DataFrame(detector.alerts_log[-10:])  # 10 log terakhir
                        log_placeholder.dataframe(df_log, use_container_width=True)
                
                # Delay kecil untuk stabilitas (tidak terlalu cepat)
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            st.error(f"Error dalam deteksi: {e}")
        finally:
            cap.release()
    
    else:
        st.info("Klik 'Mulai Deteksi' untuk memulai monitoring drowsiness")
        
    # Instructions untuk setup Telegram Bot
    with st.expander("📖 Panduan Setup Telegram Bot"):
        st.markdown("""
        ### Cara Setup Telegram Bot:
        
        1. **Buat Bot Baru:**
           - Buka Telegram dan cari `@BotFather`
           - Kirim perintah `/newbot`
           - Berikan nama dan username untuk bot Anda
           - Simpan Bot Token yang diberikan
        
        2. **Dapatkan Chat ID:**
           - Kirim pesan ke bot Anda
           - Buka: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
           - Cari `"chat":{"id":XXXXXXX}` dan salin angka tersebut
           
        3. **Untuk Group:**
           - Tambahkan bot ke group
           - Berikan admin permission
           - Chat ID group biasanya dimulai dengan tanda minus (-)
        
        4. **Test Koneksi:**
           - Masukkan Bot Token dan Chat ID di sidebar
           - Klik "Test Telegram Bot"
           - Pastikan pesan test diterima di Telegram
        """)

if __name__ == "__main__":
    main()
