## Vocalization interpretter web interface

I set this up on an Ubuntu server system in the following way:
- Install system packages listed in `requirements.system` and Python packages listed in `requirements.txt
- Copy `index.html` into to the nginx directory at `/var/www/html/`
- Copy the systemd unit file `audio_model.service` into `~/.config/systemd/user/`, and activate this as a user service using `systemctl --user enable audio_model.service && systemctl --user start audio_model.service`
