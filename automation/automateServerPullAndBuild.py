import paramiko

# Replace these variables with your own
ip_address = '62.171.179.226'
username = 'root'
password = '2q5Quvk9aJcK5B'
commands = ['cd /var/www/vhosts/jendekhane.com', 'chmod +x update.sh', './update.sh', 'pm2 restart 1']

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip_address, username, password)

# Execute commands
for command in commands:
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode('utf-8')
    print(f"Command: {command}\nOutput: {output}\n")

# Close the SSH connection
ssh.close()
